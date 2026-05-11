import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings('ignore')


torch.manual_seed(46)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

def CNN(batch_size_test=16, learning_rate=0.1, kernel_size=3, pooling='max', epochs=5, verbose=True):
    transformation = transforms.ToTensor()

    training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transformation)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transformation)


    train_loader = DataLoader(training_set, batch_size=batch_size_test, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=False)

    print(f'Downloaded {len(training_set)} images')

    if pooling == 'max':
        pool_layer = nn.MaxPool2d(kernel_size=2)
        linear_input= 2048
    elif pooling == 'avg':
        pool_layer = nn.AvgPool2d(kernel_size=2)
        linear_input = 2048
    elif pooling == 'none':
        pool_layer = nn.Identity()
        linear_input = 32768
    else:
        raise ValueError('Zły typ poolingu!')
    

    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel_size, padding=kernel_size//2),
        nn.ReLU(),
        pool_layer,
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, padding=kernel_size//2),
        nn.ReLU(),
        pool_layer,
        nn.Flatten(),
        nn.Linear(linear_input, 128),
        nn.ReLU(),
        nn.Linear(128, 10)

    ).to(device)



    criterion = nn.CrossEntropyLoss()
    optimalization = optim.Adam(model.parameters(), lr=learning_rate)

    history_loss = []

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimalization.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimalization.step()

        history_loss.append(loss.item())
        if verbose:
            print(f"Epoka {epoch+1}/{epochs} | Błąd (Loss): {loss.item():.4f}")

    class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    looking_true = 3
    looking_false = 0

    found_images = []


    model.eval()

   
    all_guesses = []
    real_results = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            diag_test = model(x)
            _, pred = torch.max(diag_test.data, 1)
            
            all_guesses.extend(pred.cpu().numpy())
            real_results.extend(y.cpu().numpy())

            for i in range(len(y)):
                if y[i].item() == looking_true and pred[i].item() == looking_false:
                    found_images.append(x[i].cpu())

                if len(found_images) >= 5:
                        break

    acc = accuracy_score(real_results, all_guesses)
    prec = precision_score(real_results, all_guesses, average='macro', zero_division=0)
    recall = recall_score(real_results, all_guesses, average='macro', zero_division=0)
    f1 = f1_score(real_results, all_guesses, average='macro', zero_division=0)
    matrix = confusion_matrix(real_results, all_guesses)


    return history_loss, acc, model, matrix, prec, recall, f1






 #Seria 1
learning_rates = [0.0001, 0.001, 0.01, 0.1]
results_loss_lr = {}
tab_lr = []
for lr in learning_rates:
    print(f'\nTesting Learning Rate: {lr}')
    loss_hist, acc, _, _, _, _, _ = CNN(batch_size_test=64, learning_rate=lr, kernel_size=3)
   
    results_loss_lr[lr] = loss_hist
    tab_lr.append({'Learning Rate': lr, 'Accuracy (%)': acc * 100})

df_lr = pd.DataFrame(tab_lr)
print('\nTabela wyników - Seria 1:')
print(df_lr.to_markdown(index=False))
plt.figure(figsize=(10,6))
for lr, loss in results_loss_lr.items():
    plt.plot(range(1, len(loss) + 1), loss, marker='^', lw=2, label=f'LR: {lr}')
plt.title('Seria 1: Wpływ Learning Rate na Błąd (Loss):')
plt.xlabel('Epoka')
plt.ylabel('Błąd (Loss)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# "Najniższy błąd (Loss) na koniec 5. epoki osiągnął model z Learning Rate = 0.001, jednak"
# " jego krzywa uczenia wykazywała początkowo oscylacje (wzrost błędu w pierwszych epokach)."
# " Zaskakująco dobrze poradziła sobie wartość 0.01, która zaprezentowała najbardziej stabilny"
# " i systematyczny spadek błędu ze wszystkich testowanych wariantów. Z kolei wartość 0.1 okazała się zbyt duża"
# " (model utknął i nie uczył się wcale), a 0.0001 uczyła sieć zbyt wolno, aby w ciągu 5 epok dogonić lepsze modele."





# #Seria 2
# batch_sizes = [16, 32, 64, 128]
# results_loss_batch = {}
# tab_batch = []

# for bs in batch_sizes:
#     print(f'\nTesting Batch Size: {bs}')

#     loss_hist, acc, model, _, _, _, _ = CNN(batch_size_test=bs, learning_rate=0.001)

#     results_loss_batch[bs] = loss_hist
#     tab_batch.append({'Batch size': bs, 'Accuracy (%)': acc * 100})

# df_batch = pd.DataFrame(tab_batch)
# print('\nTabela wyników - Seria 2:')
# print(df_batch.to_markdown(index=False))

# plt.figure(figsize=(10,6))
# for bs, loss in results_loss_batch.items():
#     plt.plot(range(1, len(loss)+1), loss, marker='o', lw=2, label=f'Batch Size: {bs}')

# plt.title('Seria 2: Wpływ Batch Size na Błąd (Loss)')
# plt.xlabel('Epoka')
# plt.ylabel('Błąd (Loss)')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()


# #Seria 3
# kernel_sizes = [3, 5, 7]
# results_loss_kernel = {}
# tab_kernel = []

# for ks in kernel_sizes:
#     print(f'Testing Kernel Size: {ks}')

#     loss_hist, acc, _, _, _, _, _ = CNN(batch_size_test=64, learning_rate=0.001, kernel_size=ks)

#     results_loss_kernel[ks] = loss_hist
#     tab_kernel.append({'Kernel Size': ks, 'Accuracy (%)': acc * 100})

# df_kernel = pd.DataFrame(tab_kernel)
# print('\nTabela wyników - Seria 3:')
# print(df_kernel.to_markdown(index=False))

# plt.figure(figsize=(10,6))
# for ks, loss in results_loss_kernel.items():
#     plt.plot(range(1, len(loss)+ 1), loss, marker='s', lw=2, label=f'Kernel Size: {ks}x{ks}')
    
# plt.title('Seria 3: Wpływ Rozmiaru Filtra (Kernel Size) na Błąd')
# plt.xlabel('Epoka')
# plt.ylabel('Błąd (Loss)')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

# "Większy rozmiar paczki danych (Batch Size: 128) sprawił, że proces uczenia był najbardziej stabilny,"
# " jednak spadek błędu następował stosunkowo wolno. Mniejsze wartości (np. 16 i 64) pozwoliły na znacznie "
# "szybszą adaptację wag i osiągnięcie najniższego błędu końcowego w ciągu 5 epok. Z kolei rozmiar 32 wykazał"
# " w tej próbie dużą niestabilność (oscylacje), co może sugerować, że przy tym rozmiarze paczki ustalony"
# " Learning Rate (0.001) był lokalnie zbyt agresywny."

# "Krzywe uczenia dla wszystkich rozmiarów filtrów wykazały dużą niestabilność (oscylacje). Choć filtr 7x7 "
# "osiągnął najniższy błąd w ostatniej epoce, jego spadek był gwałtowny i nieprzewidywalny. Pokazuje to, że w "
# "zaledwie 5 epokach większy filtr szybciej 'zapamiętał' większe wzorce, jednak w profesjonalnych sieciach preferuje "
# "się mniejsze filtry (3x3), które budują stabilniejszą hierarchię cech. "
# "Wahania wszystkich krzywych sugerują również potrzebę dalszego zmniejszenia parametru Learning Rate."



#Grid Search
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64, 128],
    'kernel_size': [3, 5, 7],
    'pooling': ['max', 'avg', 'none']
}

combinations = list(itertools.product(
    param_grid['learning_rate'],
    param_grid['batch_size'],
    param_grid['kernel_size'],
    param_grid['pooling']
))

grid_search_results = []
best_acc_global = 0
best_model_global = None
best_matrix_global = None
best_hiperparam = None

print(f'Rozpoczynanie Grid Search dla {len(combinations)} kombinacji...')

for i, (lr, bs, ks, pool) in enumerate(combinations):
    print(f'[{i+1}/{len(combinations)}] Testuję: LR={lr}, Batch={bs}, Kernel={ks}, Pool={pool}')

    loss_hist, acc, model, matrix, _, _, _ = CNN(
        learning_rate=lr,
        batch_size_test=bs,
        kernel_size=ks,
        pooling=pool,
        verbose=False
    )

    grid_search_results.append({
        'LR': lr, 'Batch': bs, 'Kernel': ks, 'Pooling': pool, 'Accuracy (%)': acc*100
    })

    if acc > best_acc_global:
        best_acc_global = acc
        best_model_global = model
        best_matrix_global = matrix
        best_hiperparam = {'LR': lr, 'Batch': bs, 'Kernel': ks, 'Pooling': pool}

df_grid = pd.DataFrame(grid_search_results)
df_grid = df_grid.sort_values(by='Accuracy (%)', ascending=False)
print('\n---TOP 5 NAJLEPSZYCH KOMBINACJI ---')
print(df_grid.head().to_markdown(index=False))

print(f'Najlepsza kombinacja: {best_hiperparam}')

#Sprawdzanie najlepszego modelu po znalezieniu go w Grid Search
loss_hist, best_acc, best_model, best_matrix, prec, recall, f1 = CNN(
    learning_rate= best_hiperparam['LR'],
    batch_size_test= best_hiperparam['Batch'],
    kernel_size= best_hiperparam['Kernel'],
    pooling= best_hiperparam['Pooling'],
    epochs= 10,
    verbose=True
)


class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print(f'Dokładność (Accuracy): {best_acc:.4f}')
print(f'Precyzja:            {prec:.4f}')
print(f'Czułość (Recall):    {recall:.4f}')
print(f'F1-Score:            {f1:.4f}')
print(f'Macierz błędu:\n{best_matrix}\n')

#Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(best_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.title('Macierz Błędu - Najlepszy Model (LR=0.001, Batch=64)', fontsize=16)
plt.xlabel('Przewidziana klasa (Machine guess)', fontsize=12)
plt.ylabel('Prawdziwa klasa (Real class)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Visualization
transformation = transforms.ToTensor()
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transformation)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

looking_true = 3
looking_false = 8
found_images = []

best_model.eval()

with torch.no_grad():
    for x, y in test_loader:
        x_gpu = x.to(device)
        diag_test =best_model(x_gpu)
        _, pred = torch.max(diag_test.data, 1)

        for i in range(len(y)):
            if y[i].item() == looking_true and pred[i].item() == looking_false:
                found_images.append(x[i].cpu())
            if len(found_images) >= 5:
                break

        if len(found_images) >= 5:
            break

if len(found_images) == 0:
    print(f'Brak pomyłek{class_names[looking_true]} jako {class_names[looking_false]}.")')
else:
    print(f"Pomyłka: {class_names[looking_true]} rozpoznany jako {class_names[looking_false]}")

    fig, axes = plt.subplots(1, len(found_images), figsize=(15, 4)) 
    if len(found_images) == 1: axes = [axes]

    for ax, img in zip(axes, found_images):

        img_to_show= img.permute(1, 2, 0).numpy()
        

        ax.imshow(img_to_show)
        ax.axis('off') 
        

        title = f"Prawda: {class_names[looking_true]}\nModel: {class_names[looking_false]}"
        ax.set_title(title, fontsize=12) 


    plt.tight_layout()
    plt.show()