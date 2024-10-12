import copy

def train_model_progressive(
    encoder,
    decoders,
    classifier,
    train_domains,
    test_domain,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    num_epochs=100,
    device="cuda",
    patience=10,
):
    print("Training model with progressive domain adaptation")
    print(f"Number of epochs: {num_epochs}")
    print(f"Patience: {patience}")
    print(f"Train domains: {train_domains}")
    print(f"Test domain: {test_domain}")
    print(f"Device: {device}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    clf_loss_fn = LabelSmoothingLoss(smoothing=0.1)
    domain_to_idx = {domain: idx for idx, domain in enumerate(train_domains + [test_domain])}

    best_loss = float("inf")
    best_test_accuracy = 0.0
    patience_counter = 0
    balancer = DynamicWeightBalancer()

    # Để lưu mô hình tốt nhất
    best_model = {
        'encoder': None,
        'decoders': None,
        'classifier': None
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        encoder.train()
        classifier.train()
        for domain in train_domains:
            decoders[domain].train()

        running_loss = 0.0
        running_recon_loss = 0.0
        running_clf_loss = 0.0
        running_kl_loss = 0.0
        total_samples = 0

        # Training loop on train dataset
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            domain_labels = torch.tensor([domain_to_idx[domain] for domain in train_domains] * (inputs.size(0) // len(train_domains)), device=device)

            mu, logvar = encoder(inputs)
            z = reparameterize(mu, logvar)

            reconstructed_imgs_list = []
            for domain in train_domains:
                domain_label = torch.tensor(
                    [domain_to_idx[domain]] * inputs.size(0), device=device
                )
                reconstructed_imgs = decoders[domain](z, domain_label)
                reconstructed_imgs_list.append(reconstructed_imgs)

            predicted_labels = classifier(z)
            predicted_domains = domain_discriminator(z.detach())  # Detach to avoid updating encoder

            loss, recon_loss, clf_loss, kl_loss, domain_loss, alpha, beta, gamma, delta = compute_loss(
                reconstructed_imgs_list,
                inputs,
                mu,
                logvar,
                predicted_labels,
                labels,
                predicted_domains,
                domain_labels,
                lambda pred, target: mixup_criterion(
                    clf_loss_fn, pred, labels_a, labels_b, lam
                ),
                domain_loss_fn,
                epoch,
                num_epochs,
                balancer,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            # Train domain discriminator
            domain_optimizer.zero_grad()
            predicted_domains = domain_discriminator(z.detach())
            domain_disc_loss = domain_loss_fn(predicted_domains, domain_labels)
            domain_disc_loss.backward()
            domain_optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_recon_loss += recon_loss * inputs.size(0)
            running_clf_loss += clf_loss * inputs.size(0)
            running_kl_loss += kl_loss * inputs.size(0)
            total_samples += inputs.size(0)

        avg_loss = running_loss / total_samples
        avg_recon_loss = running_recon_loss / total_samples
        avg_clf_loss = running_clf_loss / total_samples
        avg_kl_loss = running_kl_loss / total_samples

        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, Clf: {avg_clf_loss:.4f}, KL: {avg_kl_loss:.4f}"
        )
        print(f"Weights - Alpha: {alpha:.4f}, Beta: {beta:.4f}, Gamma: {gamma:.4f}")

        # Validation
        encoder.eval()
        classifier.eval()
        for domain in train_domains:
            decoders[domain].eval()

        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)

                mu, logvar = encoder(inputs)
                z = reparameterize(mu, logvar)

                reconstructed_imgs_list = []
                for domain in train_domains:
                    domain_label = torch.tensor(
                        [domain_to_idx[domain]] * inputs.size(0), device=device
                    )
                    reconstructed_imgs = decoders[domain](z, domain_label)
                    reconstructed_imgs_list.append(reconstructed_imgs)

                predicted_labels = classifier(z)

                val_loss, _, _, _, _, _, _ = compute_loss(
                    reconstructed_imgs_list,
                    inputs,
                    mu,
                    logvar,
                    predicted_labels,
                    labels,
                    clf_loss_fn,
                    epoch,
                    num_epochs,
                    balancer,
                )

                val_running_loss += val_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Đánh giá trên tập test
        if (epoch + 1) % 3 == 0:
            print(f"--- Evaluating on Test Domain ({test_domain}) at Epoch {epoch + 1} ---")
            test_accuracy, test_loss = evaluate_model(
                encoder,
                classifier,
                test_loader,
                device
            )
            print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")

            # Save best model based on test accuracy
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_model['encoder'] = copy.deepcopy(encoder.state_dict())
                best_model['decoders'] = {domain: copy.deepcopy(decoder.state_dict()) for domain, decoder in decoders.items()}
                best_model['classifier'] = copy.deepcopy(classifier.state_dict())
                print(f"New best model saved with test accuracy: {best_test_accuracy:.2f}%")


        # Early stopping based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        scheduler.step(avg_val_loss)

    # Load best model
    encoder.load_state_dict(best_model['encoder'])
    for domain, state_dict in best_model['decoders'].items():
        decoders[domain].load_state_dict(state_dict)
    classifier.load_state_dict(best_model['classifier'])

    print(f"Training completed. Best test accuracy: {best_test_accuracy:.2f}%")

    return encoder, decoders, classifier