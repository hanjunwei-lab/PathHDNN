import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import palettable

def train(model, input_arrray, labels, device, batch_size=64, learning_rate=0.01, lr_decay=0.5, num_epochs=10, sparse=False):
    """
    Trains a given model with specified parameters and data.

    Args:
        model: Model to be trained.
        input_array (numpy.ndarray): Input features array.
        labels (numpy.ndarray): Corresponding labels array.
        device (torch.device): Device to which tensors should be transferred ('cuda' or 'cpu').
        batch_size (int, optional): Batch size used for training. Defaults to 64.
        learning_rate (float, optional): Initial learning rate for the optimizer. Defaults to 0.01.
        lr_decay (float, optional): Decay rate for learning rate adjustment. Defaults to 0.1.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 10.
        sparse (bool, optional): If `True`, uses SGD instead of Adam for optimization. Defaults to False.

    Returns:
        tuple: Tuple containing:
            - model: Trained model.
            - train_loader (DataLoader): DataLoader for the training data.
            - test_loader (DataLoader): DataLoader for the testing data.
    """
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        input_arrray, labels, test_size=0.1, stratify=labels, random_state=42
    )
    
    # Converting numpy arrays to PyTorch tensors and moving them to the specified device
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test)

    # Creating training and testing datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Setting up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) if sparse else \
                optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=lr_decay)

    # Training the model
    model.to(device)
    model.train()
    with tqdm(range(num_epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            running_loss = 0
            i = 0
            for data, targets in train_loader:
                data , targets = data.to(device), targets.to(device)
                tepoch.set_description(f"Epoch {epoch+1}")
                
                optimizer.zero_grad()
                outputs = model(data)
                
                # Ensuring the target tensor type matches the expected by criterion
                loss = criterion(outputs, targets.float())
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                i += 1
               
            avg_loss = running_loss/i
            tepoch.set_postfix(loss=avg_loss)
            lr_scheduler.step()
    
    print('Training complete.')
    return model, train_loader, test_loader


def evaluate(model, test_loader, device):
    """
    Evaluates the performance of a trained model on a test set, given that the labels are one-hot encoded.

    Args:
        model: Trained model to be evaluated.
        test_loader (DataLoader): DataLoader containing the test data with one-hot encoded labels.

    Returns:
        str: Classification report for the evaluation.

    Examples:
        >>> model = MyModel()
        >>> test_loader = DataLoader(my_dataset, batch_size=32)
        >>> report = evaluate(model, test_loader)
        >>> print(report)
    """
    model.eval()  # Set model to eval mode to turn off dropout and batch norm
    
    y_true = []  # List to store actual class indices
    y_pred = []  # List to store predicted class indices

    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for data, labels in test_loader:
            data , labels = data.to(device), labels.to(device)
            outputs = model(data)  # Get model predictions
            _, predicted = torch.max(outputs, 1)  # Convert output probabilities to predicted class indices
            y_pred.extend(predicted.cpu().tolist())  # Append predicted class indices to y_pred list
            
            _, true = torch.max(labels, 1)  # Decode one-hot encoded labels to class indices
            y_true.extend(true.cpu().tolist())  # Append actual class indices to y_true list
    
    # Generate and return a classification report
    report = classification_report(y_true, y_pred, digits=4)
    return report

    
def get_predictions(model, data_loader):
    model.eval()  # Put the model in evaluation mode
    predictions = []
    actuals = []
    with torch.no_grad():  # No need to track gradients for evaluation
        for inputs, labels in data_loader:
            outputs = model(inputs)  # Assuming the model returns raw logits
            _, predicted = torch.max(outputs.data, 1)  # Get the index of the max logit
            predictions.extend(predicted.cpu().numpy())  # Store predictions
            _, true = torch.max(labels, 1)
            actuals.extend(true.cpu().numpy()) # Store actual labels
            
    return actuals, predictions

def plot_confusion_matrix(actuals , predictions) :
    conf_matrix = confusion_matrix(actuals, predictions , normalize='true')
    conf_matrix = np.round(conf_matrix , 2)

    fig = plt.figure(figsize=(10, 7))  # Optional, adjust size as needed
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', 
                xticklabels=sorted(np.unique(actuals)), 
                yticklabels=sorted(np.unique(actuals)))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    return fig

def get_probabilities(model, data_loader):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = torch.softmax(model(inputs), dim=1)
            probabilities.extend(nn.Softmax(dim=1)(outputs).detach().cpu().numpy())  # Assuming class "1" probabilities
            actuals.extend(labels.cpu().numpy()) # Store actual labels
            
    return actuals, probabilities

def plot_roc_curve(y_onehot_test, y_score , n_classes , target_names) : 
    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    from itertools import cycle

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(palettable.wesanderson.FantasticFox1_5.hex_colors)
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
    )
    
    return fig