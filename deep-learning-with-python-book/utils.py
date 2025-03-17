import matplotlib.pyplot as plt


def format(fig, title, xlabel, ylabel):
    color = 'white'
    
    fig.set_title(title, fontdict={'color': color, 'size': 16})
    fig.tick_params(color=color, labelcolor=color)
    fig.set_xlabel(xlabel, fontdict={'color': color, 'size': 14})
    fig.set_ylabel(ylabel, fontdict={'color': color, 'size': 14})
    

def plot(history, metric, start_ndx=0):
    # Plot the loss over epochs

    train_accuracy = history.history[metric][start_ndx:]
    val_accuracy = history.history[f'val_{metric}'][start_ndx:]

    train_loss = history.history['loss'][start_ndx:]
    val_loss = history.history['val_loss'][start_ndx:]

    epochs = range(start_ndx + 1, len(train_accuracy) + start_ndx + 1)

    plt.figure(figsize=(18, 5))

    fig1 = plt.subplot(121)
    fig1.plot(epochs, train_loss, 'g', label='Training loss')
    fig1.plot(epochs, val_loss, 'r', label='Validation loss')
    format(fig1, 'Loss', 'Epochs', 'Loss')
    fig1.legend()

    fig2 = plt.subplot(122)
    fig2.plot(epochs, train_accuracy, 'g', label=f'Training {metric}')
    fig2.plot(epochs, val_accuracy, 'r', label=f'Validation {metric}')
    format(fig2, metric, 'Epochs', metric)
    fig2.legend()
