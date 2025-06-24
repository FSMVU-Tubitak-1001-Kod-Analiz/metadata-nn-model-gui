import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import joblib
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import threading
import time

# Import your existing modules
try:
    import dnn
    import utils
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Make sure dnn.py and utils.py are in your Python path")


class FolderProcessingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Java Code Classification - Folder Processing")
        self.root.geometry("1200x900")

        # Global variables
        self.model = None
        self.scaler_code = None
        self.scaler_repo = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.class_labels = ["bug", "cleanup", "enhancement"]

        # Model and scaler paths - adjust these to your actual paths
        self.model_directory = 'models/GCB_Pooler_embeddings_content_before_only_apache_balanced_46008_embedding_reponame.pth'
        self.scaler_code_path = 'scalers/scaler_pooler_3labels_15336_embedding_reponame.pkl'  # Updated path
        # For repo name embeddings
        self.scaler_repo_path = 'scalers/scaler_pooler_3labels_15336_BERT.pkl'

        self.setup_gui()
        self.load_model_and_scalers()

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Java Code Classification Tool",
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20))

        # Folder selection frame
        folder_frame = ttk.LabelFrame(
            main_frame, text="Folder Selection", padding="10")
        folder_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        folder_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(folder_frame, text="Selected Folder:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10))

        self.folder_path_var = tk.StringVar()
        self.folder_entry = ttk.Entry(
            folder_frame, textvariable=self.folder_path_var, state="readonly")
        self.folder_entry.grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        self.browse_button = ttk.Button(
            folder_frame, text="Browse", command=self.browse_folder)
        self.browse_button.grid(row=0, column=2)

        # Info frame
        info_frame = ttk.LabelFrame(
            main_frame, text="Information", padding="10")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        info_text = ("This tool will:\n"
                     "• Scan the selected folder for Java files (.java)\n"
                     "• Process each file using content_before embedding only\n"
                     "• Use folder name as repository name\n"
                     "• Classify each file as: bug, cleanup, or enhancement")
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).grid(
            row=0, column=0, sticky=tk.W)

        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.grid_columnconfigure(2, weight=1)

        self.process_button = ttk.Button(control_frame, text="Process Folder",
                                         command=self.process_folder_threaded, state="disabled")
        self.process_button.grid(row=0, column=0, padx=(0, 10))

        self.clear_button = ttk.Button(
            control_frame, text="Clear Results", command=self.clear_results)
        self.clear_button.grid(row=0, column=1, padx=(0, 10))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(
            row=0, column=2, sticky=(tk.W, tk.E), padx=(10, 0))

        # Results frame
        results_frame = ttk.LabelFrame(
            main_frame, text="Processing Results", padding="10")
        results_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        # Results text with scrollbar
        self.results_text = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, width=100, height=30)
        self.results_text.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load model and select a folder to begin")
        status_bar = ttk.Label(
            main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

    def load_model_and_scalers(self):
        """Load the trained model and scalers"""
        try:
            self.status_var.set("Loading model and scalers...")
            self.root.update()

            # Load model using dnn.instantiate_NN_model
            self.model = dnn.instantiate_NN_model(self.model_directory)
            print("Model loaded successfully.")

            # Load scalers
            if os.path.exists(self.scaler_code_path):
                self.scaler_code = joblib.load(self.scaler_code_path)
                print(f"Code scaler loaded from: {self.scaler_code_path}")
            else:
                raise FileNotFoundError(
                    f"Code scaler not found: {self.scaler_code_path}")

            if os.path.exists(self.scaler_repo_path):
                self.scaler_repo = joblib.load(self.scaler_repo_path)
                print(f"Repo scaler loaded from: {self.scaler_repo_path}")
            else:
                raise FileNotFoundError(
                    f"Repo scaler not found: {self.scaler_repo_path}")

            self.status_var.set("Model and scalers loaded successfully")
            messagebox.showinfo(
                "Success", "Model and scalers loaded successfully!")

        except Exception as e:
            error_msg = f"Error loading model or scalers: {str(e)}"
            self.status_var.set("Error loading model")
            messagebox.showerror("Loading Error", error_msg)
            print(error_msg)

    def browse_folder(self):
        """Open folder selection dialog"""
        folder_path = filedialog.askdirectory(
            title="Select folder containing Java files")
        if folder_path:
            self.folder_path_var.set(folder_path)
            self.process_button.config(state="normal")
            self.status_var.set(f"Selected folder: {folder_path}")

    def get_java_files(self, folder_path):
        """Get all Java files from the selected folder"""
        java_files = []
        folder = Path(folder_path)

        # Search for .java files recursively
        for java_file in folder.rglob("*.java"):
            if java_file.is_file():
                java_files.append(java_file)

        return java_files

    def read_file_content(self, file_path):
        """Read content from a Java file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def prepare_dataset(self, repo_name, content_before):
        """Prepare dataset for a single file - adapted from your notebook"""
        try:
            print("Preparing dataset...")
            data = {
                'full_repo_name': [repo_name],
                'content_before': [content_before],
                'index': [0]
            }
            df = pd.DataFrame(data)

            # content_before için GraphCodeBERT pooler embedding
            print("Generating GraphCodeBERT embeddings for content_before...")
            content_embedding = utils.get_bert_embedding(
                content_before, model_type="graphcodebert")
            print(
                f"Content embedding shape after get_bert_embedding: {content_embedding.shape}")
            print(
                f"Content embedding device after get_bert_embedding: {content_embedding.device}")

            # Tensor'ı doğru şekle getir: [1, 768] -> [768]
            if content_embedding.dim() > 1:
                content_embedding = content_embedding.squeeze(
                    0)  # İlk boyutu kaldır

            print(
                f"Content embedding shape after squeeze: {content_embedding.shape}")

            # DataFrame'e eklemeden önce CPU'ya taşı
            df['embedding'] = [content_embedding.cpu()]

            # repo_name için düz BERT CLS token embedding'i
            print("Generating BERT embedding for repo_name...")
            repo_name_bert_embedding = utils.get_bert_embedding(
                repo_name, model_type="text")
            print(
                f"Repo name embedding shape after get_bert_embedding: {repo_name_bert_embedding.shape}")
            print(
                f"Repo name embedding device after get_bert_embedding: {repo_name_bert_embedding.device}")

            # Tensor'ı doğru şekle getir: [1, 768] -> [768]
            if repo_name_bert_embedding.dim() > 1:
                repo_name_bert_embedding = repo_name_bert_embedding.squeeze(
                    0)  # İlk boyutu kaldır

            print(
                f"Repo name embedding shape after squeeze: {repo_name_bert_embedding.shape}")

            # DataFrame'e eklemeden önce CPU'ya taşı
            df['repo_name_tensor'] = [repo_name_bert_embedding.cpu()]

            # Dummy label_tensor ekle
            df['label_tensor'] = [torch.tensor(
                [0, 0, 0], dtype=torch.float32).cpu()]

            print("Dataset prepared successfully.")
            return df

        except Exception as e:
            print(f"Error in prepare_dataset: {e}")
            return None

    def process_single_file(self, file_path, repo_name):
        """Process a single Java file and return prediction"""
        try:
            # Read file content
            content_before = self.read_file_content(file_path)
            if content_before is None:
                return None, "Error reading file"

            # Prepare dataset
            df_prepared = self.prepare_dataset(repo_name, content_before)
            if df_prepared is None:
                return None, "Error preparing dataset"

            print("Creating DataLoader...")
            test_loader = dnn.create_dataloader(
                df=df_prepared, batch_size=1, shuffle=False)

            print("Making prediction...")
            # dummy_criterion for test_with_data_loader
            dummy_criterion = nn.CrossEntropyLoss()

            # Use the test_with_data_loader from dnn.py which handles two scalers
            test_loss, test_accuracy, final_predicted_y, final_y = dnn.test_with_data_loader(
                self.model,
                test_loader,
                self.scaler_code,
                self.scaler_repo,
                dummy_criterion
            )

            print(f"Model Raw Outputs (logits): {final_predicted_y}")
            probabilities = torch.softmax(final_predicted_y, dim=1)
            print(f"Model Probabilities: {probabilities}")

            predicted_class_index = torch.argmax(probabilities, dim=1).item()

            # Map index to label (according to your notebook)
            idx_to_label = {0: "bug", 1: "cleanup", 2: "enhancement"}
            predicted_label = idx_to_label.get(
                predicted_class_index, "Unknown")

            # Get confidence score and all probabilities
            confidence = probabilities[0][predicted_class_index].item()
            all_probs = probabilities[0].tolist()

            # Debug info
            debug_info = f"Conf: {confidence:.3f} | Probs: bug={all_probs[0]:.3f}, cleanup={all_probs[1]:.3f}, enhancement={all_probs[2]:.3f}"

            return predicted_label, debug_info

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None, f"Error: {str(e)}"

    def process_folder_threaded(self):
        """Start folder processing in a separate thread"""
        threading.Thread(target=self.process_folder, daemon=True).start()

    def process_folder(self):
        """Process all Java files in the selected folder"""
        if not self.model or not self.scaler_code or not self.scaler_repo:
            messagebox.showerror(
                "Error", "Model and scalers must be loaded first!")
            return

        folder_path = self.folder_path_var.get()
        if not folder_path:
            messagebox.showerror("Error", "Please select a folder first!")
            return

        try:
            # Disable process button
            self.process_button.config(state="disabled")

            # Get Java files
            java_files = self.get_java_files(folder_path)

            if not java_files:
                messagebox.showwarning(
                    "No Files", "No Java files found in the selected folder!")
                self.process_button.config(state="normal")
                return

            # Clear previous results
            self.clear_results()

            # Use folder name as repo name
            repo_name = os.path.basename(folder_path)

            # Process files
            total_files = len(java_files)
            results = []

            header_text = f"Processing {total_files} Java files from: {folder_path}\n"
            header_text += f"Repository Name: {repo_name}\n"
            header_text += f"Device: {self.device}\n"
            header_text += "Loading Graph Code Bert and Bert Uncased Models...\n"
            header_text += "=" * 100 + "\n\n"

            self.results_text.insert(tk.END, header_text)
            self.results_text.update()

            start_time = time.time()

            for i, java_file in enumerate(java_files):
                # Update progress
                progress = (i / total_files) * 100
                self.progress_var.set(progress)
                self.status_var.set(
                    f"Processing file {i+1}/{total_files}: {java_file.name}")
                self.root.update()

                # Process file
                relative_path = java_file.relative_to(folder_path)
                prediction, info = self.process_single_file(
                    java_file, repo_name)

                if prediction:
                    result_text = f"[{i+1:4d}/{total_files}] {relative_path}\n"
                    result_text += f"            Prediction: {prediction} | {info}\n"
                    result_text += "-" * 80 + "\n"

                    results.append({
                        'file': str(relative_path),
                        'prediction': prediction,
                        'info': info
                    })
                else:
                    result_text = f"[{i+1:4d}/{total_files}] {relative_path}\n"
                    result_text += f"            ERROR: {info}\n"
                    result_text += "-" * 80 + "\n"

                self.results_text.insert(tk.END, result_text)
                self.results_text.see(tk.END)
                self.results_text.update()

            # Complete processing
            end_time = time.time()
            processing_time = end_time - start_time

            # Summary
            summary_text = "\n" + "=" * 100 + "\n"
            summary_text += "PROCESSING SUMMARY\n"
            summary_text += "=" * 100 + "\n"
            summary_text += f"Total files processed: {len(results)}\n"
            summary_text += f"Total files found: {total_files}\n"
            summary_text += f"Success rate: {len(results)/total_files*100:.1f}%\n"
            summary_text += f"Processing time: {processing_time:.2f} seconds\n"
            summary_text += f"Average time per file: {processing_time/total_files:.2f} seconds\n\n"

            # Count predictions
            if results:
                prediction_counts = {}
                for result in results:
                    pred = result['prediction']
                    prediction_counts[pred] = prediction_counts.get(
                        pred, 0) + 1

                summary_text += "PREDICTION DISTRIBUTION:\n"
                summary_text += "-" * 50 + "\n"
                for label in ['bug', 'cleanup', 'enhancement']:
                    count = prediction_counts.get(label, 0)
                    percentage = (count / len(results)) * 100 if results else 0
                    summary_text += f"{label:12}: {count:4d} files ({percentage:5.1f}%)\n"

                # Most common prediction
                most_common = max(prediction_counts.items(),
                                  key=lambda x: x[1])
                summary_text += f"\nMost common: {most_common[0]} ({most_common[1]} files)\n"

            self.results_text.insert(tk.END, summary_text)
            self.results_text.see(tk.END)

            # Update status and progress
            self.progress_var.set(100)
            self.status_var.set(
                f"Completed! {len(results)}/{total_files} files processed in {processing_time:.1f}s")

            messagebox.showinfo("Processing Complete",
                                f"Successfully processed {len(results)}/{total_files} Java files!\n"
                                f"Processing time: {processing_time:.1f} seconds\n"
                                f"Average: {processing_time/total_files:.2f}s per file")

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            messagebox.showerror("Processing Error", error_msg)
            self.status_var.set("Error during processing")
            print(error_msg)

        finally:
            # Re-enable process button
            self.process_button.config(state="normal")
            self.progress_var.set(0)

    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set(0)

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


# Usage
if __name__ == "__main__":
    try:
        app = FolderProcessingGUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror(
            "Startup Error", f"Error starting application: {e}")
