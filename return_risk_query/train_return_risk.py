from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "orders_dataset.csv")
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

numeric_features = ["price_inr", "discount_pct", "customer_tenure_days", "num_previous_orders", "num_previous_returns", "delivery_distance_km", "delivery_days", "is_weekend_order", "rating_given"]
categorical_features = ["product_category", "payment_method"]
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Baseline
baseline = Pipeline([("prep", preprocessor), ("model", DummyClassifier(strategy="most_frequent"))])
baseline.fit(X_train, y_train)
b_pred = baseline.predict(X_test)

# Logistic regression
lr = Pipeline([("prep", preprocessor), ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))])
lr.fit(X_train, y_train)
lr_prob = lr.predict_proba(X_test)[:, 1]
lr_pred = (lr_prob >= 0.5).astype(int)
thresholds = np.arange(0.1, 0.9001, 0.02)
lr_rows=[]
for t in thresholds:
    p=(lr_prob>=t).astype(int)
    lr_rows.append({"threshold":round(float(t),2),"f1":f1_score(y_test,p),"recall":recall_score(y_test,p),"precision":precision_score(y_test,p,zero_division=0)})
lr_sweep=pd.DataFrame(lr_rows)
lr_best=lr_sweep.loc[lr_sweep.f1.idxmax()].to_dict()

# Random forest grid search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))])
grid = GridSearchCV(rf, {"model__n_estimators":[100,200], "model__max_depth":[6,10,None]}, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
grid.fit(X_train,y_train)
best_rf=grid.best_estimator_
rf_prob=best_rf.predict_proba(X_test)[:,1]
rf_rows=[]
for t in thresholds:
    p=(rf_prob>=t).astype(int)
    rf_rows.append({"threshold":round(float(t),2),"f1":f1_score(y_test,p),"recall":recall_score(y_test,p),"precision":precision_score(y_test,p,zero_division=0)})
rf_sweep=pd.DataFrame(rf_rows)
rf_best=rf_sweep.loc[rf_sweep.f1.idxmax()].to_dict()
t_rf=float(rf_best["threshold"])

# Feature importances
feature_names=list(best_rf.named_steps["prep"].get_feature_names_out())
imp=best_rf.named_steps["model"].feature_importances_
imp_df=pd.DataFrame({"feature":feature_names,"importance":imp}).sort_values("importance",ascending=False)

def raw_feature_group(transformed):
    if transformed.startswith("num__"): return transformed[5:]
    if transformed.startswith("cat__product_category_"): return "product_category"
    if transformed.startswith("cat__payment_method_"): return "payment_method"
    return transformed
imp_df["original_feature"]=imp_df.feature.map(raw_feature_group)
raw_imp=imp_df.groupby("original_feature",as_index=False).importance.sum().sort_values("importance",ascending=False)
top5=raw_imp.head(5)["original_feature"].tolist()

# Raw-column permutation importance, using test ROC-AUC and repeating 10 times.
perm_rows=[]
rng=np.random.default_rng(42)
base_score=roc_auc_score(y_test,best_rf.predict_proba(X_test)[:,1])
for col in top5:
    drops=[]
    for _ in range(10):
        Xt=X_test.copy()
        vals=Xt[col].to_numpy().copy(); rng.shuffle(vals); Xt[col]=vals
        drops.append(base_score-roc_auc_score(y_test,best_rf.predict_proba(Xt)[:,1]))
    perm_rows.append({"feature":col,"permutation_mean_auc_drop":float(np.mean(drops))})
perm_df=pd.DataFrame(perm_rows).sort_values("permutation_mean_auc_drop",ascending=False)

# subgroup metrics
subgroups={}
for col in ["product_category","payment_method"]:
    rows=[]
    pred=(rf_prob>=t_rf).astype(int)
    tmp=X_test.copy(); tmp["y"]=y_test.to_numpy(); tmp["pred"]=pred
    for val,g in tmp.groupby(col):
        rows.append({col:val,"recall":recall_score(g.y,g.pred,zero_division=0),"precision":precision_score(g.y,g.pred,zero_division=0),"n":len(g)})
    subgroups[col]=rows

metrics={
 "dataset":{"rows":len(df),"columns":len(df.columns),"return_rate":float(y.mean()),"rating_missing_pct":float(df.rating_given.isna().mean()*100),
            "rating_missing_cod_pct":float(df.loc[df.payment_method=="COD","rating_given"].isna().mean()*100),
            "rating_missing_non_cod_pct":float(df.loc[df.payment_method!="COD","rating_given"].isna().mean()*100)},
 "return_rate_by_category":df.groupby("product_category").returned.mean().to_dict(),
 "return_rate_by_payment":df.groupby("payment_method").returned.mean().to_dict(),
 "baseline":{"accuracy":accuracy_score(y_test,b_pred),"f1_class1":f1_score(y_test,b_pred)},
 "logistic":{"accuracy":accuracy_score(y_test,lr_pred),"f1":f1_score(y_test,lr_pred),"recall":recall_score(y_test,lr_pred),"precision":precision_score(y_test,lr_pred),"roc_auc":roc_auc_score(y_test,lr_prob),"best_threshold":lr_best},
 "random_forest":{"best_params":grid.best_params_,"best_cv_roc_auc":grid.best_score_,"test_roc_auc":roc_auc_score(y_test,rf_prob),"t_rf":t_rf,"best_threshold_metrics":rf_best},
 "top5_impurity":top5,
 "top5_importance_values":raw_imp.head(5).to_dict(orient="records"),
 "permutation_top5":perm_df.to_dict(orient="records"),
 "subgroups":subgroups,
}
(MODELS/"return_risk_threshold.json").write_text(json.dumps({"t_rf":t_rf,"low_lt":t_rf,"high_gte":t_rf+0.15},indent=2))
(RESULTS/"part1_metrics.json").write_text(json.dumps(metrics,indent=2,default=lambda o:float(o)))
lr_sweep.to_csv(RESULTS/"logistic_threshold_sweep.csv",index=False)
rf_sweep.to_csv(RESULTS/"rf_threshold_sweep.csv",index=False)
raw_imp.to_csv(RESULTS/"feature_importance_by_original_feature.csv",index=False)
perm_df.to_csv(RESULTS/"permutation_importance_top5.csv",index=False)
for k,v in subgroups.items(): pd.DataFrame(v).to_csv(RESULTS/f"subgroup_{k}.csv",index=False)
joblib.dump(best_rf, MODELS/"return_risk_model.pkl")
print(json.dumps(metrics,indent=2,default=lambda o:float(o)))
