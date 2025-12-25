# predict-churn-WellCo
Predict churn for WellCo's users, a health and fitness application

Follow these steps to set up and run the project.

### 1. Clone the Repository

```bash
git clone https://github.com/Mayorov-Lior/predict-churn-WellCo.git
cd predict-churn-WellCo

### 2. install requirements
pip install -r requirements.txt
This code was run with python 3.12.7 on a Windows machine.

### 3. run code
The notebook cost_analysis.ipynb shows how to load, train and inspect the training and test data:

pc = PredictChurn(check_validation=True)
df_results_train, df_results_test = pc.train_model_and_predict()

You can set check_validation=True to test results on the validation set (random 10% of train users) or set check_validation=False to test the test set.

### Approach

Feature Engineering:

1. Combined multiple data sources: web visits, app usage, claims
2. All features were aggregated per member
3. Created time gaps statistics features from both web visits and app usage
4. for visits I added two sets of features using the page title and description -
  4.1 Number of page visits (count, unique)
  4.2 mean + std of embeddings of title + description using SentenceTransformer + reduce dimensions with TruncatedSVD
5. Added dummies of claims (given more time, I'd consider diagnosis_date)
6. I would add seniority of a member - a simple, intuitively strong feature I missed

Training:
We train models designed to rank members by churn risk instead of producing binary classifications.
We used XGBoost as the model, results aren't great. We played around a few different models and parameters but ideally would perform proper hyper parameter tuning and proper feature selection (tried, it didn't improve results but I didn't put a lot of time in that).

After running on the validation set, we turn to cost_analysis.ipynb to try and determine the best n from the validation set.
This is the thought process:
We want to minimize costs. Costs = approach costs + churn costs , both are unknown.
let's mark:
c_a = single member approach cost
c_c = single member churn cost
cost_ratio = c_a/c_c
fn = false negative, and in our case where the first n are set as positive and the est as negative, fn are all the positives that aren't in the first n. will use that for calculation.

Total Cost = n * c_a + fn * c_c
For a simpler analysis, let's divide it by c_c and get the "normalized cost":
Normalized Cost = Total Cost / c_c = n * cost_ratio + fn
=> **Normalized Cost = n * cost_ratio + fn**

In cost_analysis.ipynb we run the normalized costs as a function of n(%) for different cost ratios.
Since we have no knowledge on the cost ratio except that it's implied that it's small - we'll arbitrarily choose cost ratio = 0.2 which will set the optimal n, in our case 0.29 (29%).

We'll use that to clip our predictions on the test set to the top 29%, after calculating auc and classification report.

Our metrics:
* AUC
* Classification report
* Lift at k - calculated for the actual test results and sadly got 1.0. precision at k is 0.2986 which is equal to the churn rate at the test set (which is 10% higher that churn rate at train set, but that's how it is sometimes)


