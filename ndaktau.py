def evaluate_yolo_model(model, test_loader, iou_threshold=0.5):
    metrics = {
        'precision': [],
        'recall': [],
        'mAP': [],
        'f1': []
    }
    
    for images, targets in test_loader:
        predictions = model(images)
        
        # Hitung metrik untuk batch
        batch_metrics = calculate_batch_metrics(
            predictions, 
            targets, 
            iou_threshold
        )
        
        # Update metrics
        for key in metrics:
            metrics[key].extend(batch_metrics[key])
    
    # Hitung rata-rata metrik
    final_metrics = {
        key: np.mean(values) 
        for key, values in metrics.items()
    }
    
    return final_metrics