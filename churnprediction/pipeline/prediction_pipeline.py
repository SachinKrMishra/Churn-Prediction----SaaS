import shap
import pandas as pd
import numpy as np
from churnprediction.utils.main_utils.utils import load_object


class PredictionPipeline:

    def __init__(self):
        self.model = load_object("final_model/model.pkl")
        self.preprocessor = load_object("final_model/preprocessor.pkl")
        self.explainer = shap.TreeExplainer(self.model)

    # ---------------- CLEAN FEATURE NAME ----------------
    def clean_name(self, name):

        # Remove ColumnTransformer prefixes
        name = name.replace("num__", "").replace("cat__", "")

        # One-hot encoded handling
        if "Income_Level_" in name:
            return f"Income level ({name.split('_')[-1]})"

        if "Payment_Mode_" in name:
            return f"Payment method ({name.split('_')[-1]})"

        if "Device_Type_" in name:
            return f"Device type ({name.split('_')[-1]})"

        mapping = {
            "Age": "Customer age",
            "Number_of_Subscriptions": "Number of subscriptions",
            "Avg_Usage_Hours_Per_Week": "Weekly usage",
            "App_Switch_Frequency": "App switching behavior",
            "Discount_Used": "Discount usage",
            "Customer_Support_Interactions": "Customer support interactions",
            "Tenure_Months": "Customer tenure",
            "Monthly_Total_Spend": "Monthly spending"
        }

        return mapping.get(name, name)

    # ---------------- BUSINESS INTERPRETATION ----------------
    def interpret_impact(self, feature, impact):

        if impact > 0:
            return f"{feature} is contributing to higher churn risk"
        else:
            return f"{feature} is helping retain the customer"

    # ---------------- ACTIONS ----------------
    def generate_actions(self, reasons):

        actions = []

        for r in reasons:
            f = r["feature"]

            if "Monthly spending" in f:
                actions.append("Review pricing strategy and offer flexible plans")

            if "support interactions" in f:
                actions.append("Improve support experience and resolution speed")

            if "switching" in f:
                actions.append("Improve product stickiness and reduce friction")

            if "usage" in f:
                actions.append("Increase engagement through campaigns")

            if "discount" in f:
                actions.append("Reduce reliance on discounts and improve value")

            if "income" in f:
                actions.append("Align pricing with customer affordability")

        actions.extend([
            "Target high-risk customers proactively",
            "Run retention campaigns",
            "Monitor churn drivers continuously"
        ])

        return list(set(actions))

    # ---------------- SHAP FIX ----------------
    def process_shap(self, transformed):

        shap_values = self.explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = np.array(shap_values)

        if len(shap_values.shape) == 3:
            shap_array = shap_values[0, :, 0]
        elif len(shap_values.shape) == 2:
            shap_array = shap_values[0]
        else:
            shap_array = shap_values.flatten()

        return shap_array

    # ---------------- MAIN PREDICT ----------------
    def predict(self, input_df: pd.DataFrame):

        transformed = self.preprocessor.transform(input_df)

        prediction = self.model.predict(transformed)
        probability = self.model.predict_proba(transformed)[:, 1]

        shap_array = self.process_shap(transformed)

        feature_names = self.preprocessor.get_feature_names_out()

        min_len = min(len(shap_array), len(feature_names))
        shap_array = shap_array[:min_len]
        feature_names = feature_names[:min_len]

        reasons = []

        for i in range(min_len):

            val = float(np.ravel(shap_array[i])[0])
            raw_name = feature_names[i]

            # ---------------- FILTER ONLY ACTIVE CATEGORY ----------------
            if "Income_Level_" in raw_name:
                if input_df.iloc[0]["Income_Level"] not in raw_name:
                    continue

            if "Payment_Mode_" in raw_name:
                if input_df.iloc[0]["Payment_Mode"] not in raw_name:
                    continue

            if "Device_Type_" in raw_name:
                if input_df.iloc[0]["Device_Type"] not in raw_name:
                    continue

            human_feature = self.clean_name(raw_name)
            explanation = self.interpret_impact(human_feature, val)

            reasons.append({
                "feature": human_feature,
                "impact": val,
                "explanation": explanation
            })

        reasons = sorted(reasons, key=lambda x: abs(x["impact"]), reverse=True)

        top_positive = [r for r in reasons if r["impact"] > 0][:5]
        top_negative = [r for r in reasons if r["impact"] < 0][:5]

        actions = self.generate_actions(top_positive)

        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]

        return (
            prediction,
            probability,
            top_positive,
            top_negative,
            actions,
            float(np.ravel(base_value)[0]),
            shap_array.tolist(),
            feature_names.tolist()
        )