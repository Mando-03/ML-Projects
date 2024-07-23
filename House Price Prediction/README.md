<h1>Boston Housing Price Prediction Project</h1>
<p>This project focuses on predicting housing prices in Boston based on various features such as crime rate, property tax, accessibility to highways, etc. The dataset used for this project is <code>train.csv</code>.</p>

<h2>About the Dataset</h2>
<p>The dataset contains information about housing in Boston including:</p>
<ul>
  <li>CRIM: Per capita crime rate by town</li>
  <li>ZN: Proportion of residential land zoned for large plots</li>
  <li>INDUS: Proportion of non-retail business acres per town</li>
  <li>CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)</li>
  <li>NOX: Nitric oxides concentration (parts per 10 million)</li>
  <li>RM: Average number of rooms per dwelling</li>
  <li>AGE: Proportion of owner-occupied units built prior to 1940</li>
  <li>DIS: Weighted distances to five Boston employment centers</li>
  <li>RAD: Index of accessibility to radial highways</li>
  <li>TAX: Property tax rate per $10,000</li>
  <li>PTRATIO: Pupil-teacher ratio by town</li>
  <li>B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town</li>
  <li>LSTAT: Percentage lower status of the population</li>
  <li>MEDV: Median value of owner-occupied homes in $1000s (target variable)</li>
</ul>

<h2>Steps Taken</h2>
<h3>Data Preprocessing</h3>
<ul>
  <li><strong>Data Cleaning:</strong> Checked for missing values and handled duplicates.</li>
  <li><strong>Exploratory Data Analysis (EDA):</strong> Visualized distributions, correlations, and outliers using Matplotlib and Seaborn.</li>
  <li><strong>Feature Scaling:</strong> Applied StandardScaler to standardize the features.</li>
</ul>

<h3>Model Building</h3>
<p>Implemented and evaluated various regression algorithms for price prediction:</p>
<ul>
  <li>Linear Regression</li>
  <li>Extra Trees Regressor</li>
  <li>Ridge Regression</li>
  <li>Random Forest Regression</li>
  <li>Lasso Regression</li>
  <li>Support Vector Machine (SVM)</li>
  <li>K-Nearest Neighbors (KNN)</li>
  <li>Decision Tree</li>
  <li>Ada Boost Regressor</li>
  <li>MLPRegressor</li>
  <li>Random Forest</li>
  <li>Gradient Boosting Regressor</li>
  <li>XGBoost</li>
  <li>Stacking Regressor</li>
  <li>Voting Regressor</li>
</ul>

<h2>Results</h2>
<p>The models were evaluated based on the following metrics:</p>
<ul>
  <li><strong>R2 Score:</strong> The coefficient of determination, indicating model performance.</li>
  <li><strong>Mean Absolute Error (MAE):</strong> Average absolute difference between predicted and actual prices.</li>
  <li><strong>Mean Squared Error (MSE):</strong> Average of the squares of the differences between predicted and actual values, indicating the model's variance.</li>
  <li><strong>Root Mean Squared Error (RMSE):</strong> Square root of the average of the squares of the differences between predicted and actual values, providing a measure of the model's accuracy in the same units as the target variable.</li>
</ul>
<p>The best-performing model was Extra Trees Regressor.</p>

<h2>Hyperparameter Tuning</h2>
<p>We have done parameter tuning using GridSearchCV for:</p>
<ul>
  <li><strong>Extra Trees Regressor</strong></li>
  <li><strong>Gradient Boosting Regressor</strong></li>
</ul>

<h2>Ensemble Methods</h2>
<p>We have done Ensemble Learning using:</p>
<ul>
  <li><strong>Voting Regressor:</strong> Combined predictions from multiple models based on weighted average.</li>
  <li><strong>Stacking Regressor:</strong> Combined predictions from multiple models using a meta-learner.</li>
</ul>
<p>With Stacking Regression Outperforms others in this case so We chose Stacking Regression Model for further work.</p>

<h2>Deployment</h2>
<p>We deployed the best model using FastAPI. The following steps outline how to run the deployment:</p>



<h3>Requirements</h3>
<ul>
  <li>Python 3.7 or higher</li>
  <li>FastAPI</li>
  <li>pandas</li>
  <li>scikit-learn</li>
  <li>pydantic</li>
  <li>Uvicorn</li>
  <li>pickle</li>
  <li>NumPy</li>
</ul>
<h3>Installation</h3>
<p>Install the necessary packages using pip:</p>
<pre><code>pip install fastapi uvicorn joblib numpy</code></pre>


<h3>Running the App</h3>
<p>To run the FastAPI app, use the following command:</p>
<pre><code>uvicorn app:app --reload</code></pre>
<p>This will start the server on <code>http://127.0.0.1:8000</code>.</p>

<h3>API Endpoint</h3>
<p>The API provides an endpoint to predict housing prices:</p>
<ul>
  <li><code>/predict</code> - Accepts a JSON object with the following fields:</li>
</ul>
<pre><code>{
  "crim": 0.00632,
  "zn": 18.0,
  "indus": 2.31,
  "chas": 0,
  "nox": 0.538,
  "rm": 6.575,
  "age": 65.2,
  "dis": 4.09,
  "rad": 1,
  "tax": 296,
  "ptratio": 15.3,
  "black": 396.9,
  "lstat": 4.98
}</code></pre>
<p>The response will be a JSON object containing the predicted median value of owner-occupied homes in $1000s:</p>
<pre><code>{"medv": 21.6}</code></pre>

<h3>Example Request</h3>
<p>You can test the endpoint using a tool like <strong>curl</strong>, Postman, or directly via the provided Swagger UI by FastAPI at <code>http://127.0.0.1:8000/docs</code>:</p>
<pre><code>curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "crim": 0.00632,
  "zn": 18.0,
  "indus": 2.31,
  "chas": 0,
  "nox": 0.538,
  "rm": 6.575,
  "age": 65.2,
  "dis": 4.09,
  "rad": 1,
  "tax": 296,
  "ptratio": 15.3,
  "black": 396.9,
  "lstat": 4.98
}'</code></pre>

<h2>Files</h2>
<ul>
  <li><code>app.py</code> - Contains the FastAPI application for deploying the model.</li>
  <li><code>scaler.pkl</code> - Saved StandardScaler used for scaling input features.</li>
  <li><code>stacking_regressor_model.pkl</code> - Saved best model (Stacking Regressor).</li>
</ul>
