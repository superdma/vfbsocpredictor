# VFBSOCPredictor

#### Description
VFBSOCPredictor is a machine learning-based Vanadium Flow Battery (VFB) SOC (State of Charge) prediction project. This project uses historical cycle data (such as voltage, current, temperature, etc.), trains and predicts battery SOC status through the iTransformer time-series model, helping to optimize battery management and predict lifespan. The project is implemented in Python, supporting data preprocessing, model training, prediction, and visualization.

#### Software Architecture
The project adopts a modular structure:
- data_processing: Uses pandas to process Excel data (e.g., combined_SOC_data_20250604.xlsx)
- model: iTransformer model based on PyTorch for time-series prediction
- train_and_predict.py: Main script responsible for training and prediction, saving models as .pth files
- visualization: Uses matplotlib to generate SOC comparison charts (e.g., SOC_Comparison.png)
- results/: Stores prediction result CSV and PNG files

#### Installation

1. Clone the repository: git clone https://gitee.com/dongvma/vfbsocpredictor.git
2. Enter the directory: cd vfbsocpredictor
3. Install Python 3.8+ environment
4. Install dependencies: pip install pandas torch matplotlib scikit-learn openpyxl (inferred from project files)
5. Verify installation: python -c "import pandas, torch; print('Installation successful')"

#### Instructions

1. Prepare data: Place cycle data in the root directory (e.g., cycle79-80.xlsx)
2. Train model: python train_and_predict.py --train (using combined_SOC_data_20250604.xlsx for training)
3. Perform prediction: python train_and_predict.py --predict --cycle <cycle_num> (e.g., Cycle_2)
4. View results: The results/ folder contains prediction CSV files and PNG comparison charts
5. Customize: Modify draw_picture.py to adjust visualization parameters, or view learning rate curves

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme_XXX.md to support different languages, such as Readme_en.md, Readme_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
