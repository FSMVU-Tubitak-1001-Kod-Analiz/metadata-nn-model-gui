# Java Code Classification Tool - Metadata Neural Network Model GUI

A comprehensive GUI-based tool for classifying Java code changes into three categories: **bug fixes**, **cleanup**, and **enhancements**. The tool supports both Deep Neural Network (DNN) and traditional Machine Learning models.

*This project is part of the FSMVU-Tubitak-1001 Code Analysis research initiative.*

## 🚀 Features

- **Multi-Model Support**: Choose from 10 different classification models including DNN, XGBoost, Random Forest, and more
- **GitHub Integration**: Direct repository download and processing
- **Interactive GUI**: User-friendly interface with tree view results
- **Batch Processing**: Process entire folders of Java files at once
- **Real-time Progress**: Live progress tracking and detailed logging
- **Export Results**: Save classification results as CSV files
- **Visual Analytics**: Summary statistics and confidence scores

## 📋 Requirements

### System Requirements
- **Python 3.12.3** (tested and verified - exact version recommended)
- Torch with CUDA compatibility
- CUDA-compatible GPU (optional, for DNN model acceleration)
- Git (for repository downloading)
- 8GB+ RAM recommended
- Windows 10/11, macOS, or Linux

### Python Dependencies
**Tested with Python 3.12.3 - Exact versions from working environment:**

```bash
# Core ML libraries
torch==2.5.1
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.6.1
joblib==1.4.2

# Machine Learning models
xgboost==3.0.2
catboost==1.2.7
lightgbm==4.6.0

# NLP and embeddings
transformers==4.51.3
huggingface-hub==0.31.1
tokenizers==0.21.0

# PyTorch ecosystem
torchaudio==2.5.1
torchvision==0.20.1

# Additional utilities
matplotlib==3.10.0
seaborn==0.13.2
plotly==6.1.2
tqdm==4.67.1
psutil==5.9.0
```

## 🛠️ Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/FSMVU-Tubitak-1001-Kod-Analiz/metadata-nn-model-gui.git
cd metadata-nn-model-gui
```

### Step 2: Create Virtual Environment (Recommended)

**Option A: Using Conda (Recommended - exact environment)**
```bash
# Create environment from provided file
conda env create -f environment.yml
conda activate java-classifier
```

**Option B: Using pip with virtual environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

**If using Conda (from Step 2A):**
Dependencies are already installed with the environment.

**If using pip (from Step 2B):**
```bash
# Install exact tested versions
pip install -r requirements.txt
```

**Alternative - Install specific Python version first:**
```bash
# If you don't have Python 3.12.3, install it first
# Then create virtual environment and install dependencies
python3.12 -m venv venv
# ... activate and pip install
```

### Step 4: Download Pre-trained Models

#### ML Models (Required)
Download the pre-trained ML models from: [**Google Drive - ML Models**](https://drive.google.com/file/d/1GfHIcwDHM4k0xxGrFe0k8Ah-zT3V5Lb2/view?usp=sharing)

**Download Steps:**
1. Click the Google Drive link above
2. Click "Download" (you may need to click "Download anyway" if Google shows a warning)
3. Extract the downloaded `ml_models.zip` file
4. Place the extracted `ml_models` folder in the project root directory

Extract the downloaded file and place the `ml_models` folder in the project root directory:
```
metadata-nn-model-gui/
├── ml_models/
│   ├── XGBoost_model.pkl
│   ├── RandomForest_model.pkl
│   ├── CatBoost_model.pkl
│   ├── LightGBM_model.pkl
│   ├── KNN_model.pkl
│   ├── NaiveBayes_model.pkl
│   ├── LogisticRegression_model.pkl
│   ├── AdaBoost_model.pkl
│   └── SVM_model.pkl
├── models/ (already included)
├── scalers/ (already included)
└── gui_main.py
```

#### DNN Model (Already Included)
The DNN model and scalers are already included in the repository under:
- `models/` - Contains the pre-trained DNN model
- `scalers/` - Contains the feature scalers

### Step 5: Verify Installation
```bash
# Activate your environment first
conda activate java-classifier  # if using conda
# OR
source venv/bin/activate  # if using pip on macOS/Linux
# OR  
venv\Scripts\activate  # if using pip on Windows

# Run the application
python gui_main.py
```

**Check Python version:**
```bash
python --version
# Should output: Python 3.12.3
```

If the GUI opens successfully, you're ready to go!

## 📖 Usage Guide

### Starting the Application
```bash
python gui_main.py
```

### Basic Workflow

1. **Select a Model**
   - Choose from 10 available models in the dropdown
   - Click "Load Model" to initialize
   - Wait for "Model loaded successfully" message

2. **Prepare Your Data**
   - **Option A**: Download from GitHub
     - Enter a GitHub repository URL
     - Click "Download Repository"
     - The tool will clone the repo to `data_to_try/` folder
   
   - **Option B**: Use Local Folder
     - Click "Browse" to select a folder containing Java files
     - The tool will recursively find all `.java` files

3. **Process Files**
   - Click "Process Folder" to start classification
   - Monitor progress in the progress bar
   - View real-time results in the "Detailed Log" tab

4. **Analyze Results**
   - **File Structure Tab**: Tree view of all files with predictions
   - **Summary Tab**: Statistics and distribution charts
   - **Detailed Log Tab**: Complete processing log

5. **Export Results**
   - Click "Export Results" to save as CSV
   - Results include file paths, predictions, and confidence scores

### Model Information

| Model | Type | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| DNN | Deep Learning | Slow | High | Complex patterns |
| XGBoost | Gradient Boosting | Fast | High | General purpose |
| Random Forest | Ensemble | Fast | Good | Stable predictions |
| CatBoost | Gradient Boosting | Medium | High | Categorical features |
| LightGBM | Gradient Boosting | Very Fast | High | Large datasets |
| SVM | Support Vector | Medium | Good | Small datasets |
| Logistic Regression | Linear | Very Fast | Medium | Baseline |
| KNN | Instance-based | Fast | Medium | Simple patterns |
| Naive Bayes | Probabilistic | Very Fast | Medium | Text classification |
| AdaBoost | Ensemble | Medium | Good | Weak learners |

## 🎯 Classification Categories

The tool classifies Java code changes into three categories:

- **🐛 Bug**: Code changes that fix bugs or errors
- **🧹 Cleanup**: Code refactoring, formatting, or cleanup changes
- **✨ Enhancement**: New features or improvements

## 🔧 Configuration

### Model Paths
You can modify model paths in `gui_main.py`:

```python
# DNN model paths
self.dnn_model_directory = 'models/your_dnn_model.pth'
self.scaler_code_path = 'scalers/your_code_scaler.pkl'
self.scaler_repo_path = 'scalers/your_repo_scaler.pkl'

# ML models directory
self.ml_models_directory = 'ml_models'
```

### Device Selection
The tool automatically detects CUDA availability:
- GPU: Used for DNN model (faster)
- CPU: Used for ML models and fallback

## 📁 Project Structure

```
java-code-classifier/
├── gui_main.py           # Main GUI application
├── dnn.py               # DNN model utilities
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── models/             # DNN model files
│   └── GCB_Pooler_embeddings_content_before_only_apache_balanced_46008_embedding_reponame.pth
├── scalers/            # Feature scalers
│   ├── scaler_pooler_3labels_15336_embedding_reponame.pkl
│   └── scaler_pooler_3labels_15336_BERT.pkl
├── ml_models/          # ML model files (download separately)
│   ├── XGBoost_model.pkl
│   ├── RandomForest_model.pkl
│   └── ... (other models)
└── data_to_try/        # Downloaded repositories (auto-created)
```

## 🐛 Troubleshooting

### Common Issues

**1. Python Version Issues**
```bash
# Check your Python version
python --version

# Should be Python 3.12.3 for best compatibility
# If different version, consider using conda:
conda install python=3.12.3
```

**2. Module Import Errors**
```bash
# Make sure all dependencies are installed with exact versions
pip install -r requirements.txt

# Or use conda environment
conda env create -f environment.yml
conda activate java-classifier

# Verify Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**2. CUDA Out of Memory**
```python
# Edit gui_main.py to force CPU usage
self.device = 'cpu'  # Force CPU usage
```

**3. Model Loading Errors**
- Verify model files exist in correct directories
- Check file permissions
- Ensure models were downloaded completely
- Verify you're using Python 3.12.3

**4. Git Clone Errors**
- Install Git: https://git-scm.com/downloads
- Check repository URL format
- Verify internet connection

**5. GUI Not Opening**
```bash
# On Linux, install tkinter
sudo apt-get install python3-tk

# On macOS with brew
brew install python-tk

# On Windows, tkinter should be included with Python
```

**6. Version Conflicts**
```bash
# If you have conflicting package versions:
pip freeze | grep -E "(torch|numpy|pandas)"

# Uninstall and reinstall with exact versions:
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

### Performance Tips

1. **For Large Repositories**:
   - Use ML models (faster than DNN)
   - Process in smaller batches
   - Ensure sufficient RAM

2. **For Better Accuracy**:
   - Use DNN model
   - Ensure GPU availability
   - Use repositories similar to training data

3. **For Speed**:
   - Use LightGBM or XGBoost
   - Process on SSD storage
   - Close other applications

## 📊 Example Output

```
PROCESSING SUMMARY
==================================================

📁 Repository: spring-framework
🔧 Model: XGBoost
💻 Device: CPU

📊 STATISTICS:
   • Total files found: 1,247
   • Successfully processed: 1,244
   • Success rate: 99.8%
   • Processing time: 45.23 seconds
   • Average per file: 0.04 seconds

🎯 PREDICTION DISTRIBUTION:
   🐛 Bug         :  423 files (34.0%) ██████▊
   🧹 Cleanup     :  298 files (24.0%) ████▊
   ✨ Enhancement :  523 files (42.0%) ████████▍

🏆 Most common: enhancement (523 files)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Your License Here]

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/FSMVU-Tubitak-1001-Kod-Analiz/metadata-nn-model-gui/issues)
- **Documentation**: [Wiki](https://github.com/FSMVU-Tubitak-1001-Kod-Analiz/metadata-nn-model-gui/wiki)
- **Email**: Contact via GitHub Issues

## 🏷️ Version History

- **v1.0.0**: Initial release
  - Multi-model support
  - GUI interface
  - GitHub integration
  - Export functionality

---

**Made with ❤️ for Java developers and researchers**
