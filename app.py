import os
import uuid
import base64
from io import BytesIO
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import shapiro

app = Flask(__name__)

# Configuration for upload and plot folders
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_y(input_features, model, poly_features):
    """
    Predict Y using the polynomial regression model.
    """
    X_new_poly = poly_features.transform([input_features])
    Y_pred = model.predict(X_new_poly)
    return Y_pred[0]

def generate_regression_equation(feature_coefficients, intercept=0.0):
    """
    Generate the regression equation as a string.
    """
    equation = f"Y = {intercept:.4f}"
    for feature, coef in feature_coefficients:
        if coef >= 0:
            equation += f" + ({coef:.4f})*{feature}"
        else:
            equation += f" - ({abs(coef):.4f})*{feature}"
    return equation

def generate_rsm_plots(model, poly_features, feature_names, X, plot_folder):
    """
    Generate RSM (Response Surface Methodology) plots for each pair of features.
    """
    num_features = len(feature_names)
    plots = []
    
    # Define pairs of features for RSM plots
    feature_pairs = [(0,1), (0,2), (1,2)]  # Adjust based on number of features
    
    for pair in feature_pairs:
        i, j = pair
        feature_x = feature_names[i]
        feature_y = feature_names[j]
        
        # Create grid
        x_min, x_max = X[:, i].min(), X[:, i].max()
        y_min, y_max = X[:, j].min(), X[:, j].max()
        x_range = np.linspace(x_min, x_max, 100)
        y_range = np.linspace(y_min, y_max, 100)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        
        # Keep other features constant (mean value)
        X_const = np.mean(X, axis=0)
        
        # Prepare input for prediction
        Z = np.zeros_like(X_grid)
        for idx in range(len(X_grid.ravel())):
            row = X_const.copy()
            row[i] = X_grid.ravel()[idx]
            row[j] = Y_grid.ravel()[idx]
            Z.ravel()[idx] = model.predict(poly_features.transform([row]))[0]
        
        # Reshape Z to grid shape
        Z = Z.reshape(X_grid.shape)
        
        # Plotting
        fig = plt.figure(figsize=(14, 6))
        
        # 3D Surface Plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(X_grid, Y_grid, Z, cmap='autumn_r', alpha=0.7)
        ax1.set_xlabel(feature_x)
        ax1.set_ylabel(feature_y)
        ax1.set_zlabel('Predicted Y')
        ax1.set_title(f'3D Surface: {feature_x} vs {feature_y}')
        
        # Contour Plot
        ax2 = fig.add_subplot(1, 2, 2)
        contour = ax2.contourf(X_grid, Y_grid, Z, cmap='autumn_r', alpha=0.7)
        ax2.contour(X_grid, Y_grid, Z, colors='white', linewidths=1)
        ax2.set_xlabel(feature_x)
        ax2.set_ylabel(feature_y)
        ax2.set_title(f'Contour: {feature_x} vs {feature_y}')
        fig.colorbar(contour, ax=ax2, shrink=0.6)
        
        # Save plot to file
        plot_filename = f'rsm_{feature_x}_{feature_y}_{uuid.uuid4().hex}.png'
        plot_path = os.path.join(plot_folder, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        # Append plot filename to list
        plots.append(plot_filename)
    
    return plots

# Initialize global variables
reg = None
poly_features = None
X_global = None
feature_names_global = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global reg, poly_features, X_global, feature_names_global  # Declare globals
    
    if 'excel_file' not in request.files:
        return redirect(request.url)
    
    file = request.files['excel_file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process the Excel file
        try:
            data = pd.read_excel(filepath)
            # Data validation
            if data.isnull().values.any():
                return "Data contains missing values. Please complete your data.", 400
            
            if data.shape[1] < 2:
                return "Excel file must have at least two columns (features and target).", 400
            
            # Ensure exactly three feature columns for prediction
            if data.shape[1] != 4:  # 3 features + 1 target
                return "Excel file must have exactly three feature columns and one target column.", 400
            
            # Ensure all feature columns are numeric
            feature_cols = data.columns[:-1]
            target_col = data.columns[-1]
            
            if not data[feature_cols].apply(pd.api.types.is_numeric_dtype).all().all():
                return "All feature columns must be numeric.", 400
            
            # Ensure target column is numeric
            if not pd.api.types.is_numeric_dtype(data[target_col]):
                return "Target column must be numeric.", 400
            
            # Prepare data for the model
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            
            feature_names = data.columns[:-1].tolist()
            target_name = data.columns[-1]
            
            # Store globally for prediction
            X_global = X
            feature_names_global = feature_names
            
            # Generate polynomial features
            degree = 2
            poly_features = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly_features.fit_transform(X)
            
            # Train the model
            reg = LinearRegression()
            reg.fit(X_poly, y)
            
            # Get feature names from PolynomialFeatures
            poly_feature_names = poly_features.get_feature_names_out(input_features=feature_names)
            
            # Combine feature names with coefficients (excluding bias term)
            feature_coefficients = list(zip(poly_feature_names, reg.coef_))
            intercept = reg.intercept_
            
            # Generate the regression equation string
            regression_equation = generate_regression_equation(feature_coefficients, intercept=intercept)
            
            # Generate RSM plots
            rsm_plots = generate_rsm_plots(reg, poly_features, feature_names, X, app.config['PLOT_FOLDER'])
            
            # Predictions and Residuals
            y_pred = reg.predict(X_poly)
            residuals = y - y_pred
            
            # Plot Residuals
            plt.figure(figsize=(10,6))
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(0, color='red', linestyle='--')
            plt.title('Residual Plot')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_residual_png = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close()
            
            # Histogram of Residuals
            plt.figure(figsize=(10,6))
            plt.hist(residuals, bins=30, color='gray', edgecolor='black')
            plt.title('Histogram of Residuals')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_histogram_png = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close()
            
            # Normality test (Shapiro-Wilk)
            _, p_value = shapiro(residuals)
            
            # Convert DataFrame to HTML table
            data_html = data.to_html(classes='table table-striped', index=False)
            
            # Render the result template
            return render_template('result.html',
                                   feature_coefficients=feature_coefficients,
                                   intercept=intercept,
                                   regression_equation=regression_equation,
                                   r_squared=reg.score(X_poly, y),
                                   plot_images=rsm_plots,
                                   data_table=data_html,
                                   plot_residuals=plot_residual_png,
                                   plot_histogram=plot_histogram_png,
                                   p_value=p_value)
            
        except Exception as e:
            return f"An error occurred while processing the file: {str(e)}", 500
    else:
        return "Invalid file format. Please upload an Excel file with .xlsx or .xls extension.", 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if reg is None or poly_features is None:
            return "Model has not been trained yet. Please upload and process data first.", 400
        
        # Retrieve input values
        input_values = []
        for i in range(1, 4):  # Assuming three features
            value = request.form.get(f'x{i}')
            if value is None:
                return f"Missing input for feature x{i}.", 400
            try:
                input_values.append(float(value))
            except ValueError:
                return f"Invalid input for feature x{i}. Please enter numeric values.", 400
        
        # Predict Y
        predicted_Y = predict_y(input_values, reg, poly_features)
        
        return render_template('prediction.html', predicted_Y=predicted_Y)
    except Exception as e:
        return f"Error during prediction: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
