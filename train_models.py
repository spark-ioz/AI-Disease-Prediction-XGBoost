
"""
Optional XGBoost demonstration training script.

IMPORTANT:
This script creates SYNTHETIC educational data. It is NOT a clinical dataset and
the resulting models must NOT be presented as medically validated predictors.
No clinical accuracy claim is made.

The web app itself remains usable without these model files.
"""
import os, numpy as np, joblib
from xgboost import XGBClassifier

BASE=os.path.dirname(os.path.abspath(__file__))
MODEL_DIR=os.path.join(BASE,"models")
os.makedirs(MODEL_DIR,exist_ok=True)

rng=np.random.default_rng(42)
n=2000
age=rng.integers(18,80,n)
bmi=np.clip(rng.normal(25,4.5,n),15,45)
sbp=np.clip(rng.normal(122,18,n),85,190)
dbp=np.clip(rng.normal(78,10,n),45,120)
hr=np.clip(rng.normal(75,12,n),45,130)
temp=np.clip(rng.normal(98.2,0.8,n),95,104)
sleep=np.clip(rng.normal(7,1.4,n),3,12)
water=np.clip(rng.normal(2.0,0.7,n),0.2,5)
activity=np.clip(rng.normal(35,25,n),0,180)
stress=rng.integers(1,11,n)
smoking=rng.integers(0,2,n)

X=np.column_stack([age,bmi,sbp,dbp,hr,temp,sleep,water,activity,stress,smoking])
# Synthetic labels for demonstration only.
diab=((bmi>30)*0.35+(age>45)*0.2+rng.random(n)*0.45>0.55).astype(int)
hyper=((sbp>135)*0.5+(dbp>85)*0.3+(age>50)*0.1+rng.random(n)*0.25>0.5).astype(int)
heart=((sbp>140)*0.3+(hr>100)*0.2+smoking*0.2+(age>50)*0.2+rng.random(n)*0.3>0.5).astype(int)
resp=((smoking*0.35)+(rng.random(n)*0.4)+0.2>0.55).astype(int)

features=["Age","BMI","SystolicBP","DiastolicBP","HeartRate","Temperature","SleepHours","WaterLitres","ActivityMinutes","StressLevel","SmokingStatus"]
joblib.dump(features,os.path.join(MODEL_DIR,"feature_columns.joblib"))
for name,y in [("diabetes",diab),("hypertension",hyper),("heart",heart),("respiratory",resp)]:
    model=XGBClassifier(n_estimators=120,max_depth=3,learning_rate=0.06,subsample=0.9,colsample_bytree=0.9,eval_metric="logloss",random_state=42)
    model.fit(X,y)
    joblib.dump(model,os.path.join(MODEL_DIR,f"{name}_xgboost.joblib"))
print("Synthetic demonstration XGBoost models created.")
print("Do not use these models for clinical decisions or claim clinical accuracy.")
