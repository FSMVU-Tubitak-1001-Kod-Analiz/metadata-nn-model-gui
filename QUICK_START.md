# 🚀 Quick Start Guide

Get up and running with the Java Code Classification Tool in 5 minutes!

## ⚡ TL;DR Setup

**Option A: Using Conda (Recommended)**
```bash
# 1. Clone and enter directory
git clone https://github.com/FSMVU-Tubitak-1001-Kod-Analiz/metadata-nn-model-gui.git
cd metadata-nn-model-gui

# 2. Create conda environment with exact dependencies
conda env create -f environment.yml
conda activate java-classifier

# 3. Download ML models from Google Drive and extract to ml_models/ folder

# 4. Run the application
python gui_main.py
```

**Option B: Using pip**
```bash
# 1. Clone and enter directory
git clone https://github.com/FSMVU-Tubitak-1001-Kod-Analiz/metadata-nn-model-gui.git
cd metadata-nn-model-gui

# 2. Create virtual environment (requires Python 3.12.3)
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 3. Install exact tested versions
pip install -r requirements.txt

# 4. Download ML models from Google Drive and extract to ml_models/ folder

# 5. Run the application
python gui_main.py
```

## 📱 Quick Usage

### Step 1: Load a Model
1. Open the application
2. Select a model from dropdown (try **XGBoost** for best balance)
3. Click **"Load Model"**
4. Wait for green "Model loaded successfully" message

### Step 2: Get Java Code
**Option A - Download from GitHub:**
1. Paste any GitHub repo URL (e.g., `https://github.com/spring-projects/spring-framework`)
2. Click **"Download Repository"**
3. Click **"Yes"** when asked to use the folder

**Option B - Use Local Folder:**
1. Click **"Browse"**
2. Select any folder containing `.java` files

### Step 3: Process & View Results
1. Click **"Process Folder"**
2. Watch the progress bar
3. View results in the **"File Structure"** tab
4. Check **"Summary"** for statistics

## 🎯 What You'll Get

- **🐛 Bug fixes**: Code that fixes errors
- **🧹 Cleanup**: Refactoring and code cleanup  
- **✨ Enhancements**: New features and improvements

Each prediction includes confidence scores and detailed logs.

## ⚙️ Model Recommendations

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **First time user** | XGBoost | Fast, accurate, reliable |
| **Best accuracy** | DNN | Highest quality predictions |
| **Fastest processing** | LightGBM | Speed optimized |
| **Small datasets** | Random Forest | Stable on small data |
| **Research/Analysis** | DNN | Most sophisticated |

## 🔧 Common Issues & Solutions

**"Module not found" error:**
```bash
# Option A: Use conda environment
conda env create -f environment.yml
conda activate java-classifier

# Option B: Install exact versions
pip install -r requirements.txt

# Option C: Check Python version
python --version  # Should be 3.12.3 for best compatibility
```

**GUI won't open on Linux:**
```bash
sudo apt-get install python3-tk
```

**Out of memory:**
- Use ML models instead of DNN
- Process smaller folders
- Close other applications

**Slow processing:**
- Try LightGBM or XGBoost models
- Use SSD storage
- Ensure sufficient RAM

## 📁 Example Workflow

1. **Load XGBoost model** (recommended for beginners)
2. **Download a repository**: `https://github.com/spring-projects/spring-boot`
3. **Process the repository** (may take 2-5 minutes for large repos)
4. **Check results**:
   - File Structure: See individual file predictions
   - Summary: View overall statistics
   - Export: Save results as CSV

## 💡 Pro Tips

- **Start small**: Test with small repositories first
- **Compare models**: Try different models on the same data
- **Export results**: Save CSV files for further analysis
- **Use filters**: Right-click in tree view for options
- **Check confidence**: Higher confidence = more reliable prediction

## 🆘 Need Help?

- Check the full [README.md](README.md) for detailed instructions
- Open an [issue](https://github.com/FSMVU-Tubitak-1001-Kod-Analiz/metadata-nn-model-gui/issues) on GitHub
- Contact via GitHub Issues

---

**Happy classifying! 🎉**