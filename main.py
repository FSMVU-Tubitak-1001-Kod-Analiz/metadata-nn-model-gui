import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import PhotoImage
import threading
import time
import joblib
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

from utils import prepare_dataframe, load_embeddings_from_directory_for_df
from DNN import onehot_embed_df, instantiate_NN_model, create_dataloader, test_with_data_loader

# Global variables for animation and style
angle = 0
processing_complete = False
font ="Helvetica"

# Function to handle directory selection
def select_directory():
    global selected_directory
    directory_path = filedialog.askdirectory(title="Select a Dataset Directory")
    if directory_path:  # If a directory is selected
        selected_directory = directory_path
        directory_label.config(text=f"Selected Directory:\n{directory_path}")
        prepare_button.config(state='normal')
    else:
        selected_directory = None
        directory_label.config(text="No directory selected")
        prepare_button.config(state='disabled')


def show_confusion_matrix():
    # Compute confusion matrix
    _, predicted = torch.max(test_outputs.detach().cpu(), 1)  # Get the index of the max log-probability
    labels_1d = torch.argmax(y_test, dim=1).cpu()  # Turn into 1D tensor
    
    # Convert to numpy arrays
    y_test_np = labels_1d.numpy()
    predicted_np = predicted.numpy()
    
    cm = confusion_matrix(y_test_np, predicted_np)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot the confusion matrix with matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    ax.set_title('Confusion Matrix for Multi-Class Classification')

    # Show the graph separately
    plt.show()

# 
def display_prediction_results():
    global test_outputs, onehot_df  # Access global variables

    # Ensure the predictions are available
    if test_outputs is None:
        details_textbox.insert("end", "No predictions available.\n")
        return

    # Convert predictions and true labels to numpy for easy manipulation
    _, predicted_indices = torch.max(test_outputs.detach().cpu(), 1)
    predicted_indices_np = predicted_indices.numpy()

    # Original DataFrame should have a 'file_path' and 'label' (if necessary)
    original_df = onehot_df#.reset_index()  # Reset index if necessary for alignment

    # Iterate and map predictions back to file paths and display them
    details_textbox.delete("1.0", "end")  # Clear previous content
    for i, prediction in enumerate(predicted_indices_np):
        file_path = original_df.loc[i, 'file_path']  # Assuming 'file_path' column exists
        predicted_label = prediction  # This might map to an actual label via a dictionary

        # If labels are class indices, map them to class names (optional)
        label_mapping = {0: "Enhancement", 1: "Bug", 2: "Cleanup"}
        predicted_label_text = label_mapping.get(predicted_label, f"Class {predicted_label}")

        # Format the output and insert it into the text box
        output_text = f"File: {file_path}\nPrediction: {predicted_label_text}\n\n"
        details_textbox.insert("end", output_text)

# Function to animate the loading spinner
def animate_processing():
    global angle, processing_complete
    if not processing_complete:
        processing_canvas.delete("all")
        x0, y0, x1, y1 = 15, 15, 50, 50
        processing_canvas.create_arc(x0, y0, x1, y1, start=angle, width=5, extent=40, style=tk.ARC)
        angle += 20
        if angle >= 360:
            angle = 0
        root.after(50, animate_processing)
    else:
        # Display a green tick
        processing_canvas.delete("all")
        processing_canvas.create_line(15, 30, 30, 45, fill="green", width=5)
        processing_canvas.create_line(30, 45, 45, 15, fill="green", width=5)

# Function to prepare the dataset
def prepare_dataset():
    global onehot_df
    
    if not selected_directory:
        raise ValueError("No directory selected! Please select a directory.")

    # Prepare the raw dataframe
    df = prepare_dataframe(file_path="merged.csv", should_balance=False, should_merge_csv=False)

    # Recursively find all CSV files in the selected directory
    csv_files = []
    for root, dirs, files in os.walk(selected_directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))  # Store full file path

    if not csv_files:
        raise ValueError("No CSV files found in the selected directory!")

    # Extract repository names from the directory structure and filenames
    repo_names = set()
    for file_path in csv_files:
        # Extract the parent directory as org_name
        org_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        # Extract the repo_name from the filename (removing _pr_data.csv)
        repo_name = os.path.basename(os.path.dirname(file_path))
        # Combine them in the format "org_name/repo_name"
        full_repo_name = f"{org_name}/{repo_name}"
        repo_names.add(full_repo_name)

    if not repo_names:
        raise ValueError("No valid repository names found in the CSV filenames!")

    # One-hot encode the raw dataframe
    onehot_df = onehot_embed_df(df=df)

    # Filter the one-hot-encoded dataframe based on the repository names
    filtered_df = onehot_df[onehot_df['full_repo_name'].isin(repo_names)].drop(columns=['full_repo_name','file_path'])

    # Load embeddings for the filtered dataframe
    final_df = load_embeddings_from_directory_for_df(
        directory_path="Embeddings/GCB_Pooler_embeddings_patch_before_largewithcodedata_Xlimit3label", df=filtered_df)

    print(final_df)
    return final_df

# Function to prepare the model
def prepare_model(df):
    return instantiate_NN_model(df=df, model_directory=
                         "Models/two_pipelines_StScaler_pooler_6layers_invertednn_100epoch_batch_normalization.pth") 

# Thread worker to prepare dataset and model
def prepare_worker():
    global processing_complete, model, test_loader
    try:
        evaluate_button.config(state="disabled")
        prepare_button.config(state="disabled")

        # Prepare dataset
        df = prepare_dataset()

        # Prepare model
        model = prepare_model(df=df)

        # Prepare dataloader
        batch_size = 128
        test_loader = create_dataloader(df=df, batch_size=batch_size)

        # Signal completion
        processing_complete = True
        evaluate_button.config(state="normal")
        prepare_button.config(state="normal")
    except Exception as e:
        processing_complete = True
        print(f"Error during preparation: {e}")

# Button callback to start the preparation process
def on_prepare():
    global processing_complete
    processing_complete = False

    # Start the animation
    animate_processing()

    # Run the preparation in a separate thread
    threading.Thread(target=prepare_worker).start()

# Function to start the testing process
def on_evaluate():
    global test_outputs, y_test
    criterion = nn.CrossEntropyLoss()
    scaler = joblib.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Scalers/scaler_pooler_3labels_15300.pkl'))

    with torch.no_grad():
        model.eval()
        test_loss, test_accuracy, test_outputs, y_test = test_with_data_loader(model=model, loader=test_loader,
                                                                               scaler=scaler, criterion=criterion)
    results_label.config(text=f'Results:\nAccuracy on test data: {test_accuracy:.4f}\nLoss value on test data: {test_loss:.4f}')

    confusion_matrix_button.config(state="active")
    
    display_prediction_results()

# Main gui code
if __name__ == "__main__":
    root = tk.Tk()
    root.title("TUBITAK Code1001 - Code Label Predictor")
    root.geometry("900x800")  # Adjusted for a wider layout
    model_list = ["Deep Neural Network", "XGBoost", "RandomForest", "CatBoost", "LightGBM", 
                  "KNN", "NaiveBayes", "LogisticRegression", "AdaBoost", "SVM"]

    # Create the main 3-column layout with a frame for each column
    left_frame = tk.Frame(root)
    left_frame.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

    middle_frame = tk.Frame(root)
    middle_frame.grid(row=0, column=1, sticky="n", padx=10, pady=10)

    right_frame = tk.Frame(root)
    right_frame.grid(row=0, column=2, sticky="ne", padx=10, pady=10)

    # LEFT COLUMN: Logo
    logo_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Logo/fsm100x100.png')
    logo_image = tk.PhotoImage(file=logo_path)
    logo_label = tk.Label(left_frame, image=logo_image)
    logo_label.pack()

    # RIGHT COLUMN: Loading Animation
    processing_canvas = tk.Canvas(right_frame, width=65, height=65, highlightthickness=0)
    processing_canvas.pack()

    # MIDDLE COLUMN: Main GUI Elements
    # Title Label
    title_label = tk.Label(middle_frame, text="Code Label Predictor", font=(font, 25))
    title_label.pack(pady=40)

    # Model Selection and Directory Selector
    frame_top = tk.Frame(middle_frame)
    frame_top.pack(pady=5)

    model_label = tk.Label(frame_top, text="Model:", font=(font, 15, "bold"))
    model_label.grid(row=0, column=0, padx=5)
    model_combobox = ttk.Combobox(frame_top, font=(font, 15, "bold"), values=model_list)
    model_combobox.grid(row=0, column=1, padx=15)
    model_combobox.set(model_list[0])

    directory_button = tk.Button(frame_top, text="Select Sample Directory", command=select_directory, font=(font, 15, "bold"))
    directory_button.grid(row=1, column=0, columnspan=2, pady=5)

    # Directory Label
    directory_label = tk.Label(
        frame_top, 
        text="No directory selected", 
        font=(font, 12, "italic"), 
        wraplength=400,  
        justify="left"
    )
    directory_label.grid(row=2, column=0, columnspan=2, pady=5)

    # Prepare Button
    prepare_button = tk.Button(middle_frame, state='disabled', command=on_prepare, text="Prepare Dataset & Sample", font=(font, 15, "bold"))
    prepare_button.pack(pady=5)

    # Evaluate Button
    evaluate_button = tk.Button(middle_frame, state="disabled", command=on_evaluate, text="Evaluate", font=(font, 15, "bold"))
    evaluate_button.pack(pady=5)

    # Results Section
    results_label = tk.Label(middle_frame, text="Results:\n", font=(font, 15, "bold"), justify="left")
    results_label.pack(pady=10, fill="x")

    # Confusion Matrix Button
    confusion_matrix_button = tk.Button(middle_frame, state="disabled", command=show_confusion_matrix, text="Show Confusion Matrix", font=(font, 15, "bold"))
    confusion_matrix_button.pack(pady=10)

    # "Result Details" Label
    details_label = tk.Label(middle_frame, text="Result Details:", font=(font, 15, "bold"))
    details_label.pack(pady=5, anchor="w")

    # Frame for Text Widget and Scrollbar
    details_frame = tk.Frame(middle_frame)
    details_frame.pack(padx=10, pady=5, fill="both", expand=True)

    # Text Widget for Result Details
    details_textbox = tk.Text(details_frame, wrap="word", font=(font, 12), height=15, width=80)
    details_textbox.pack(side="left", fill="both", expand=True)

    # Scrollbar for Text Widget
    scrollbar = tk.Scrollbar(details_frame, orient="vertical", command=details_textbox.yview)
    scrollbar.pack(side="right", fill="y")

    # Configure Scrollbar for Text Widget
    details_textbox.config(yscrollcommand=scrollbar.set)

    root.mainloop()
