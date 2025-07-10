import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import threading
import time
import subprocess
import shutil
import re

# Import your existing modules
try:
    import DNN
    import utils
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Make sure dnn.py and utils.py are in your Python path")


class FolderProcessingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Java Code Classification - Folder Processing")
        self.root.geometry("1400x1000")

        # Global variables
        self.model = None
        self.ml_model = None  # For ML models
        self.current_model_type = None  # 'dnn' or 'ml'
        self.scaler_code = None
        self.scaler_repo = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.class_labels = ["bug", "cleanup", "enhancement"]

        # Model and scaler paths - adjust these to your actual paths
        self.dnn_model_directory = 'models/GCB_Pooler_embeddings_content_before_only_apache_balanced_46008_embedding_reponame.pth'
        self.scaler_code_path = 'scalers/scaler_pooler_3labels_15336_embedding_reponame.pkl'
        self.scaler_repo_path = 'scalers/scaler_pooler_3labels_15336_BERT.pkl'

        # ML models directory
        self.ml_models_directory = 'ml_models'

        # Available models
        self.available_models = {
            'DNN (Deep Neural Network)': 'dnn',
            'XGBoost': 'ml_models/XGBoost_model.pkl',
            'Random Forest': 'ml_models/RandomForest_model.pkl',
            'CatBoost': 'ml_models/CatBoost_model.pkl',
            'LightGBM': 'ml_models/LightGBM_model.pkl',
            'KNN': 'ml_models/KNN_model.pkl',
            'Naive Bayes': 'ml_models/NaiveBayes_model.pkl',
            'Logistic Regression': 'ml_models/LogisticRegression_model.pkl',
            'AdaBoost': 'ml_models/AdaBoost_model.pkl',
            'SVM': 'ml_models/SVM_model.pkl'
        }

        # TreeView için renkler
        self.prediction_colors = {
            'bug': '#ffebee',        # Light red
            'cleanup': '#fff3e0',    # Light orange
            'enhancement': '#e8f5e8',  # Light green
            'error': '#ffcccb',      # Light pink for errors
            'folder': '#f0f0f0'      # Light gray for folders
        }

        self.setup_gui()

    def setup_gui(self):
        # Ana container - Paned Window kullanarak bölümleri ayıralım
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Sol panel - Kontroller
        left_frame = ttk.Frame(main_paned, width=500)
        main_paned.add(left_frame, weight=1)

        # Sağ panel - Sonuçlar
        right_frame = ttk.Frame(main_paned, width=900)
        main_paned.add(right_frame, weight=2)

        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)

    def setup_left_panel(self, parent):
        """Sol paneli kur - kontroller"""
        # Title
        title_label = ttk.Label(parent, text="Java Code Classification Tool",
                                font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))

        # GitHub Repository Download Frame
        github_frame = ttk.LabelFrame(
            parent, text="GitHub Repository Download", padding="10")
        github_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(github_frame, text="GitHub URL:").pack(anchor=tk.W)

        self.github_url_var = tk.StringVar()
        self.github_entry = ttk.Entry(
            github_frame, textvariable=self.github_url_var)
        self.github_entry.pack(fill=tk.X, pady=(5, 5))

        self.download_button = ttk.Button(github_frame, text="Download Repository",
                                          command=self.download_github_repo_threaded)
        self.download_button.pack()

        # Download status
        self.download_status_var = tk.StringVar()
        self.download_status_label = ttk.Label(
            github_frame, textvariable=self.download_status_var)
        self.download_status_label.pack(pady=(5, 0))

        # Model selection frame
        model_frame = ttk.LabelFrame(
            parent, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="Select Model:").pack(anchor=tk.W)

        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(model_frame, textvariable=self.model_var,
                                           values=list(
                                               self.available_models.keys()),
                                           state="readonly")
        self.model_combobox.pack(fill=tk.X, pady=(5, 5))
        self.model_combobox.set(
            'DNN (Deep Neural Network)')  # Default selection

        self.load_model_button = ttk.Button(model_frame, text="Load Model",
                                            command=self.load_selected_model)
        self.load_model_button.pack()

        # Model status
        self.model_status_var = tk.StringVar()
        self.model_status_var.set("No model loaded")
        ttk.Label(model_frame, textvariable=self.model_status_var,
                  foreground="red").pack(pady=(5, 0))

        # Folder selection frame
        folder_frame = ttk.LabelFrame(
            parent, text="Folder Selection", padding="10")
        folder_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(folder_frame, text="Selected Folder:").pack(anchor=tk.W)

        self.folder_path_var = tk.StringVar()
        self.folder_entry = ttk.Entry(
            folder_frame, textvariable=self.folder_path_var, state="readonly")
        self.folder_entry.pack(fill=tk.X, pady=(5, 5))

        self.browse_button = ttk.Button(
            folder_frame, text="Browse", command=self.browse_folder)
        self.browse_button.pack()

        # Info frame
        info_frame = ttk.LabelFrame(parent, text="Information", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        info_text = ("This tool will:\n"
                     "• Download GitHub repositories\n"
                     "• Scan folder for Java files\n"
                     "• Classify as: bug, cleanup, enhancement\n"
                     "• Show results in tree structure")
        ttk.Label(info_frame, text=info_text,
                  justify=tk.LEFT).pack(anchor=tk.W)

        # Control buttons frame
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)

        self.process_button = ttk.Button(button_frame, text="Process Folder",
                                         command=self.process_folder_threaded, state="disabled")
        self.process_button.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_button = ttk.Button(
            button_frame, text="Clear Results", command=self.clear_results)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 5))

        self.export_button = ttk.Button(
            button_frame, text="Export Results", command=self.export_results)
        self.export_button.pack(side=tk.LEFT)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(
            "Ready - Select and load a model, then choose a folder to begin")
        status_bar = ttk.Label(
            parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def setup_right_panel(self, parent):
        """Sağ paneli kur - sonuçlar"""
        # Results frame
        results_frame = ttk.LabelFrame(
            parent, text="Processing Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # TreeView için notebook (tabbed interface)
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tree View Tab
        tree_frame = ttk.Frame(self.notebook)
        self.notebook.add(tree_frame, text="File Structure")

        # TreeView oluştur
        self.setup_treeview(tree_frame)

        # Summary Tab
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")

        # Summary için scrolled text
        self.summary_text = scrolledtext.ScrolledText(
            summary_frame, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # Details Tab
        details_frame = ttk.Frame(self.notebook)
        self.notebook.add(details_frame, text="Detailed Log")

        # Details için scrolled text
        self.details_text = scrolledtext.ScrolledText(
            details_frame, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True)

    def setup_treeview(self, parent):
        """TreeView'i kur"""
        # TreeView container
        tree_container = ttk.Frame(parent)
        tree_container.pack(fill=tk.BOTH, expand=True)

        # TreeView columns
        columns = ('prediction', 'confidence', 'details')
        self.tree = ttk.Treeview(
            tree_container, columns=columns, show='tree headings')

        # Column headings
        self.tree.heading('#0', text='File Path', anchor=tk.W)
        self.tree.heading('prediction', text='Prediction', anchor=tk.CENTER)
        self.tree.heading('confidence', text='Confidence', anchor=tk.CENTER)
        self.tree.heading('details', text='Details', anchor=tk.W)

        # Column widths
        self.tree.column('#0', width=300, minwidth=200)
        self.tree.column('prediction', width=100, minwidth=80)
        self.tree.column('confidence', width=80, minwidth=60)
        self.tree.column('details', width=200, minwidth=150)

        # Scrollbars
        tree_scrollbar_y = ttk.Scrollbar(
            tree_container, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scrollbar_x = ttk.Scrollbar(
            tree_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_scrollbar_y.set,
                            xscrollcommand=tree_scrollbar_x.set)

        # Pack TreeView ve scrollbar'lar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        # TreeView event bindings
        self.tree.bind('<Double-1>', self.on_tree_double_click)
        # Right click menu
        self.tree.bind('<Button-3>', self.on_tree_right_click)

        # Context menu
        self.create_context_menu()

        # Configure tags for colors
        for pred_type, color in self.prediction_colors.items():
            self.tree.tag_configure(pred_type, background=color)

    def create_context_menu(self):
        """Context menu oluştur"""
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(
            label="Copy Path", command=self.copy_selected_path)
        self.context_menu.add_command(
            label="Show Details", command=self.show_selected_details)
        self.context_menu.add_separator()
        self.context_menu.add_command(
            label="Expand All", command=self.expand_all)
        self.context_menu.add_command(
            label="Collapse All", command=self.collapse_all)

    def on_tree_double_click(self, event):
        """TreeView double click event"""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            self.show_item_details(item)

    def on_tree_right_click(self, event):
        """TreeView right click event"""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)

    def copy_selected_path(self):
        """Seçili path'i kopyala"""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            path = self.get_item_full_path(item)
            self.root.clipboard_clear()
            self.root.clipboard_append(path)
            messagebox.showinfo("Copied", f"Path copied to clipboard:\n{path}")

    def get_item_full_path(self, item):
        """TreeView item'inin tam path'ini al"""
        path_parts = []
        current = item

        while current:
            text = self.tree.item(current)['text']
            if text and text != 'Root':
                path_parts.append(text)
            current = self.tree.parent(current)

        return '/'.join(reversed(path_parts))

    def show_selected_details(self):
        """Seçili item'in detaylarını göster"""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            self.show_item_details(item)

    def show_item_details(self, item):
        """Item detaylarını popup'ta göster"""
        item_data = self.tree.item(item)
        text = item_data['text']
        values = item_data['values']

        if values:  # Sadece dosyalar için
            details = f"File: {text}\n\n"
            details += f"Prediction: {values[0]}\n"
            details += f"Confidence: {values[1]}\n"
            details += f"Details: {values[2]}\n"
            details += f"Full Path: {self.get_item_full_path(item)}"

            messagebox.showinfo("File Details", details)

    def expand_all(self):
        """Tüm tree'yi genişlet"""
        def expand_recursive(item):
            self.tree.item(item, open=True)
            for child in self.tree.get_children(item):
                expand_recursive(child)

        for item in self.tree.get_children():
            expand_recursive(item)

    def collapse_all(self):
        """Tüm tree'yi kapat"""
        def collapse_recursive(item):
            self.tree.item(item, open=False)
            for child in self.tree.get_children(item):
                collapse_recursive(child)

        for item in self.tree.get_children():
            collapse_recursive(item)

    def clear_results(self):
        """Sonuçları temizle"""
        # TreeView'i temizle
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Text alanlarını temizle
        self.summary_text.delete(1.0, tk.END)
        self.details_text.delete(1.0, tk.END)

        self.progress_var.set(0)

    def create_folder_structure_in_tree(self, files_data, root_folder_name):
        """TreeView'de klasör yapısını oluştur"""
        # Root node ekle
        root_item = self.tree.insert(
            '', 'end', text=root_folder_name, open=True, tags=['folder'])

        # Klasör yapısını oluşturmak için gerekli veri yapısı
        folder_items = {root_folder_name: root_item}

        for file_data in files_data:
            relative_path = file_data['file']
            prediction = file_data.get('prediction', 'error')
            info = file_data.get('info', 'No info')

            # Path'i parçalara ayır
            path_parts = str(relative_path).split(os.sep)

            # Her klasör seviyesi için item oluştur
            current_path = root_folder_name
            parent_item = root_item

            for i, part in enumerate(path_parts):
                current_path = os.path.join(
                    current_path, part) if current_path != root_folder_name else os.path.join(root_folder_name, part)

                if i == len(path_parts) - 1:  # Son part = dosya
                    # Confidence bilgisini parse et
                    confidence = "N/A"
                    if "Conf:" in info:
                        try:
                            conf_part = info.split("Conf:")[1].split("|")[
                                0].strip()
                            confidence = conf_part
                        except:
                            confidence = "N/A"

                    # Dosya item'ini ekle
                    file_item = self.tree.insert(parent_item, 'end',
                                                 text=part,
                                                 values=(
                                                     prediction, confidence, info),
                                                 tags=[prediction])
                else:  # Klasör
                    if current_path not in folder_items:
                        folder_item = self.tree.insert(parent_item, 'end',
                                                       text=part,
                                                       open=True,
                                                       tags=['folder'])
                        folder_items[current_path] = folder_item
                    parent_item = folder_items[current_path]

    def update_summary_tab(self, results, processing_time, total_files):
        """Summary tab'ını güncelle"""
        summary_text = "PROCESSING SUMMARY\n"
        summary_text += "=" * 50 + "\n\n"
        summary_text += f"📁 Repository: {os.path.basename(self.folder_path_var.get())}\n"
        summary_text += f"🔧 Model: {self.model_var.get()}\n"
        summary_text += f"💻 Device: {self.device if self.current_model_type == 'dnn' else 'CPU'}\n\n"

        summary_text += f"📊 STATISTICS:\n"
        summary_text += f"   • Total files found: {total_files}\n"
        summary_text += f"   • Successfully processed: {len(results)}\n"
        summary_text += f"   • Success rate: {len(results)/total_files*100:.1f}%\n"
        summary_text += f"   • Processing time: {processing_time:.2f} seconds\n"
        summary_text += f"   • Average per file: {processing_time/total_files:.2f} seconds\n\n"

        if results:
            prediction_counts = {}
            for result in results:
                pred = result['prediction']
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1

            summary_text += "🎯 PREDICTION DISTRIBUTION:\n"
            for label in ['bug', 'cleanup', 'enhancement']:
                count = prediction_counts.get(label, 0)
                percentage = (count / len(results)) * 100 if results else 0
                # Emoji için
                emoji = "🐛" if label == 'bug' else "🧹" if label == 'cleanup' else "✨"
                bar = "█" * int(percentage / 5)  # Simple bar chart
                summary_text += f"   {emoji} {label.capitalize():12}: {count:4d} files ({percentage:5.1f}%) {bar}\n"

            most_common = max(prediction_counts.items(), key=lambda x: x[1])
            summary_text += f"\n🏆 Most common: {most_common[0]} ({most_common[1]} files)\n"

        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary_text)

    def export_results(self):
        """Sonuçları CSV olarak export et"""
        if not self.tree.get_children():
            messagebox.showwarning("No Data", "No results to export!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Results"
        )

        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        ['File Path', 'Prediction', 'Confidence', 'Details'])

                    def export_item(item, path=""):
                        item_data = self.tree.item(item)
                        text = item_data['text']
                        values = item_data['values']

                        current_path = os.path.join(
                            path, text) if path else text

                        if values:  # Dosya
                            writer.writerow(
                                [current_path, values[0], values[1], values[2]])

                        # Alt öğeleri işle
                        for child in self.tree.get_children(item):
                            export_item(child, current_path)

                    for root_item in self.tree.get_children():
                        for child in self.tree.get_children(root_item):
                            export_item(child, "")

                messagebox.showinfo("Export Complete",
                                    f"Results exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror(
                    "Export Error", f"Error exporting results:\n{str(e)}")

    # Buradan sonra mevcut metodlar aynı kalacak, sadece process_folder metodunda değişiklik var

    def load_selected_model(self):
        """Load the selected model"""
        selected = self.model_var.get()
        if not selected:
            messagebox.showerror("Error", "Please select a model first!")
            return

        model_path = self.available_models[selected]

        if model_path == 'dnn':
            self.load_dnn_model()
        else:
            self.load_ml_model(model_path, selected)

    def load_dnn_model(self):
        """Load the DNN model and scalers"""
        try:
            self.status_var.set("Loading DNN model and scalers...")
            self.root.update()

            # Load model using dnn.instantiate_NN_model
            self.model = DNN.instantiate_NN_model(self.dnn_model_directory)
            self.current_model_type = 'dnn'
            print("DNN model loaded successfully.")

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

            self.model_status_var.set("DNN model loaded successfully")
            self.status_var.set("DNN model and scalers loaded successfully")
            self.check_ready_to_process()
            messagebox.showinfo(
                "Success", "DNN model and scalers loaded successfully!")

        except Exception as e:
            error_msg = f"Error loading DNN model or scalers: {str(e)}"
            self.status_var.set("Error loading DNN model")
            self.model_status_var.set("Failed to load model")
            messagebox.showerror("Loading Error", error_msg)
            print(error_msg)

    def load_ml_model(self, model_path, model_name):
        """Load an ML model"""
        try:
            self.status_var.set(f"Loading {model_name}...")
            self.root.update()

            if os.path.exists(model_path):
                self.ml_model = joblib.load(model_path)
                self.current_model_type = 'ml'
                self.model_status_var.set(f"{model_name} loaded successfully")
                self.status_var.set(f"{model_name} loaded successfully")
                self.check_ready_to_process()
                messagebox.showinfo(
                    "Success", f"{model_name} loaded successfully!")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

        except Exception as e:
            error_msg = f"Error loading {model_name}: {str(e)}"
            self.status_var.set(f"Error loading {model_name}")
            self.model_status_var.set("Failed to load model")
            messagebox.showerror("Loading Error", error_msg)
            print(error_msg)

    def check_ready_to_process(self):
        """Check if we're ready to process files"""
        if self.folder_path_var.get() and (self.model or self.ml_model):
            self.process_button.config(state="normal")
        else:
            self.process_button.config(state="disabled")

    def browse_folder(self):
        """Open folder selection dialog"""
        folder_path = filedialog.askdirectory(
            title="Select folder containing Java files")
        if folder_path:
            self.folder_path_var.set(folder_path)
            self.check_ready_to_process()
            self.status_var.set(f"Selected folder: {folder_path}")

    def get_java_files(self, folder_path):
        """Get all Java files from the selected folder"""
        java_files = []
        folder = Path(folder_path)
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

    def prepare_dataset_for_dnn(self, repo_name, content_before):
        """Prepare dataset for DNN model"""
        try:
            print("Preparing dataset for DNN...")
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

            if content_embedding.dim() > 1:
                content_embedding = content_embedding.squeeze(0)

            df['embedding'] = [content_embedding.cpu()]

            # repo_name için düz BERT CLS token embedding'i
            print("Generating BERT embedding for repo_name...")
            repo_name_bert_embedding = utils.get_bert_embedding(
                repo_name, model_type="text")

            if repo_name_bert_embedding.dim() > 1:
                repo_name_bert_embedding = repo_name_bert_embedding.squeeze(0)

            df['repo_name_tensor'] = [repo_name_bert_embedding.cpu()]
            df['label_tensor'] = [torch.tensor(
                [0, 0, 0], dtype=torch.float32).cpu()]

            print("Dataset prepared successfully for DNN.")
            return df

        except Exception as e:
            print(f"Error in prepare_dataset_for_dnn: {e}")
            return None

    def prepare_features_for_ml(self, repo_name, content_before):
        """Prepare features for ML models"""
        try:
            print("Preparing features for ML model...")

            # Get embeddings
            content_embedding = utils.get_bert_embedding(
                content_before, model_type="graphcodebert")
            if content_embedding.dim() > 1:
                content_embedding = content_embedding.squeeze(0)

            repo_embedding = utils.get_bert_embedding(
                repo_name, model_type="text")
            if repo_embedding.dim() > 1:
                repo_embedding = repo_embedding.squeeze(0)

            # Combine embeddings
            combined_features = np.concatenate([
                content_embedding.cpu().numpy(),
                repo_embedding.cpu().numpy()
            ])

            # Reshape for single prediction
            return combined_features.reshape(1, -1)

        except Exception as e:
            print(f"Error in prepare_features_for_ml: {e}")
            return None

    def process_single_file_dnn(self, file_path, repo_name):
        """Process a single file using DNN model"""
        try:
            content_before = self.read_file_content(file_path)
            if content_before is None:
                return None, "Error reading file"

            df_prepared = self.prepare_dataset_for_dnn(
                repo_name, content_before)
            if df_prepared is None:
                return None, "Error preparing dataset"

            print("Creating DataLoader...")
            test_loader = DNN.create_dataloader(
                df=df_prepared, batch_size=1, shuffle=False)

            print("Making prediction with DNN...")
            dummy_criterion = nn.CrossEntropyLoss()

            test_loss, test_accuracy, final_predicted_y, final_y = DNN.test_with_data_loader(
                self.model,
                test_loader,
                self.scaler_code,
                self.scaler_repo,
                dummy_criterion
            )

            probabilities = torch.softmax(final_predicted_y, dim=1)
            predicted_class_index = torch.argmax(probabilities, dim=1).item()

            idx_to_label = {0: "bug", 1: "cleanup", 2: "enhancement"}
            predicted_label = idx_to_label.get(
                predicted_class_index, "Unknown")

            confidence = probabilities[0][predicted_class_index].item()
            all_probs = probabilities[0].tolist()

            debug_info = f"Conf: {confidence:.3f} | Probs: bug={all_probs[0]:.3f}, cleanup={all_probs[1]:.3f}, enhancement={all_probs[2]:.3f}"

            return predicted_label, debug_info

        except Exception as e:
            print(f"Error processing file with DNN {file_path}: {e}")
            return None, f"Error: {str(e)}"

    def process_single_file_ml(self, file_path, repo_name):
        """Process a single file using ML model"""
        try:
            content_before = self.read_file_content(file_path)
            if content_before is None:
                return None, "Error reading file"

            features = self.prepare_features_for_ml(repo_name, content_before)
            if features is None:
                return None, "Error preparing features"

            print("Making prediction with ML model...")
            prediction = self.ml_model.predict(features)

            # If the model supports predict_proba, get probabilities
            try:
                probabilities = self.ml_model.predict_proba(features)[0]
                confidence = probabilities[prediction[0]]
                debug_info = f"Conf: {confidence:.3f} | Probs: bug={probabilities[0]:.3f}, cleanup={probabilities[1]:.3f}, enhancement={probabilities[2]:.3f}"
            except:
                debug_info = "No probability scores available"

            idx_to_label = {0: "bug", 1: "cleanup", 2: "enhancement"}
            predicted_label = idx_to_label.get(prediction[0], "Unknown")

            return predicted_label, debug_info

        except Exception as e:
            print(f"Error processing file with ML {file_path}: {e}")
            return None, f"Error: {str(e)}"

    def process_single_file(self, file_path, repo_name):
        """Process a single file based on current model type"""
        if self.current_model_type == 'dnn':
            return self.process_single_file_dnn(file_path, repo_name)
        else:
            return self.process_single_file_ml(file_path, repo_name)

    def process_folder_threaded(self):
        """Start folder processing in a separate thread"""
        threading.Thread(target=self.process_folder, daemon=True).start()

    def process_folder(self):
        """Process all Java files in the selected folder"""
        if self.current_model_type == 'dnn':
            if not self.model or not self.scaler_code or not self.scaler_repo:
                messagebox.showerror(
                    "Error", "DNN model and scalers must be loaded first!")
                return
        elif self.current_model_type == 'ml':
            if not self.ml_model:
                messagebox.showerror("Error", "ML model must be loaded first!")
                return
        else:
            messagebox.showerror("Error", "No model loaded!")
            return

        folder_path = self.folder_path_var.get()
        if not folder_path:
            messagebox.showerror("Error", "Please select a folder first!")
            return

        try:
            self.process_button.config(state="disabled")
            java_files = self.get_java_files(folder_path)

            if not java_files:
                messagebox.showwarning(
                    "No Files", "No Java files found in the selected folder!")
                self.process_button.config(state="normal")
                return

            self.clear_results()
            repo_name = os.path.basename(folder_path)

            total_files = len(java_files)
            results = []

            model_type = self.model_var.get()

            # Details tab'ına başlık ekle
            header_text = f"Processing {total_files} Java files from: {folder_path}\n"
            header_text += f"Repository Name: {repo_name}\n"
            header_text += f"Model: {model_type}\n"
            header_text += f"Device: {self.device if self.current_model_type == 'dnn' else 'CPU'}\n"
            header_text += "=" * 100 + "\n\n"

            self.details_text.insert(tk.END, header_text)
            self.details_text.update()

            start_time = time.time()

            for i, java_file in enumerate(java_files):
                progress = (i / total_files) * 100
                self.progress_var.set(progress)
                self.status_var.set(
                    f"Processing file {i+1}/{total_files}: {java_file.name}")
                self.root.update()

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

                    results.append({
                        'file': str(relative_path),
                        'prediction': 'error',
                        'info': info
                    })

                self.details_text.insert(tk.END, result_text)
                self.details_text.see(tk.END)
                self.details_text.update()

            end_time = time.time()
            processing_time = end_time - start_time

            # TreeView'e sonuçları ekle
            self.create_folder_structure_in_tree(results, repo_name)

            # Summary tab'ını güncelle
            self.update_summary_tab(results, processing_time, total_files)

            # Details tab'ına özet ekle
            summary_text = "\n" + "=" * 100 + "\n"
            summary_text += "PROCESSING COMPLETE\n"
            summary_text += "=" * 100 + "\n"
            summary_text += f"Total files processed: {len([r for r in results if r['prediction'] != 'error'])}\n"
            summary_text += f"Total files found: {total_files}\n"
            success_count = len(
                [r for r in results if r['prediction'] != 'error'])
            summary_text += f"Success rate: {success_count/total_files*100:.1f}%\n"
            summary_text += f"Processing time: {processing_time:.2f} seconds\n"
            summary_text += f"Average time per file: {processing_time/total_files:.2f} seconds\n"

            self.details_text.insert(tk.END, summary_text)
            self.details_text.see(tk.END)

            self.progress_var.set(100)
            self.status_var.set(
                f"Completed! {success_count}/{total_files} files processed in {processing_time:.1f}s")

            # TreeView'de ilk tab'ı seç
            self.notebook.select(0)

            messagebox.showinfo("Processing Complete",
                                f"Successfully processed {success_count}/{total_files} Java files!\n"
                                f"Processing time: {processing_time:.1f} seconds\n"
                                f"Average: {processing_time/total_files:.2f}s per file\n\n"
                                f"Check the File Structure tab for detailed results!")

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            messagebox.showerror("Processing Error", error_msg)
            self.status_var.set("Error during processing")
            print(error_msg)

        finally:
            self.process_button.config(state="normal")
            self.progress_var.set(0)

    def download_github_repo_threaded(self):
        """Download GitHub repo in a separate thread"""
        threading.Thread(target=self.download_github_repo, daemon=True).start()

    def download_github_repo(self):
        """Download a GitHub repository"""
        github_url = self.github_url_var.get().strip()

        if not github_url:
            messagebox.showerror(
                "Error", "Please enter a GitHub repository URL!")
            return

        # Validate GitHub URL
        github_pattern = r'https://github\.com/[\w-]+/[\w-]+'
        if not re.match(github_pattern, github_url):
            messagebox.showerror(
                "Error", "Invalid GitHub URL! Please enter a valid repository URL.")
            return

        try:
            # Create data_to_try directory if it doesn't exist
            data_dir = Path("data_to_try")
            data_dir.mkdir(exist_ok=True)

            # Extract repository name from URL
            repo_name = github_url.rstrip('/').split('/')[-1]
            repo_owner = github_url.rstrip('/').split('/')[-2]

            # Update status
            self.download_status_var.set(
                f"Downloading {repo_owner}/{repo_name}...")
            self.download_button.config(state="disabled")
            self.root.update()

            # Target directory
            target_dir = data_dir / repo_name

            # If directory exists, ask user
            if target_dir.exists():
                response = messagebox.askyesno(
                    "Directory Exists",
                    f"The repository '{repo_name}' already exists. Do you want to replace it?"
                )
                if response:
                    shutil.rmtree(target_dir)
                else:
                    self.download_status_var.set("Download cancelled")
                    self.download_button.config(state="normal")
                    return

            # Clone the repository
            try:
                # Check if git is installed
                subprocess.run(["git", "--version"],
                               check=True, capture_output=True)

                # Clone the repository
                result = subprocess.run(
                    ["git", "clone", github_url, str(target_dir)],
                    capture_output=True,
                    text=True,
                    check=True
                )

                self.download_status_var.set(
                    f"Successfully downloaded {repo_name}")

                # Ask if user wants to use this folder
                response = messagebox.askyesno(
                    "Download Complete",
                    f"Repository downloaded to: {target_dir}\n\nDo you want to use this folder for processing?"
                )

                if response:
                    self.folder_path_var.set(str(target_dir))
                    self.check_ready_to_process()
                    self.status_var.set(f"Selected folder: {target_dir}")

            except subprocess.CalledProcessError as e:
                if "git" in str(e):
                    messagebox.showerror(
                        "Git Not Found",
                        "Git is not installed or not in PATH.\nPlease install Git to download repositories."
                    )
                else:
                    messagebox.showerror(
                        "Clone Error", f"Failed to clone repository:\n{e.stderr}")
                self.download_status_var.set("Download failed")

        except Exception as e:
            messagebox.showerror(
                "Download Error", f"Error downloading repository:\n{str(e)}")
            self.download_status_var.set("Download failed")

        finally:
            self.download_button.config(state="normal")

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
