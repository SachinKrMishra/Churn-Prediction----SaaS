import sys
import pandas as pd

from churnprediction.exception.exception import ChurnPredictionException


class CustomData:
    def __init__(
        self,
        Age: int,
        Number_of_Subscriptions: int,
        Avg_Usage_Hours_Per_Week: float,
        App_Switch_Frequency: int,
        Discount_Used: int,
        Customer_Support_Interactions: int,
        Tenure_Months: int,
        Monthly_Total_Spend: float,
        Income_Level: str,
        Payment_Mode: str,
        Device_Type: str
    ):
        try:
            self.Age = Age
            self.Number_of_Subscriptions = Number_of_Subscriptions
            self.Avg_Usage_Hours_Per_Week = Avg_Usage_Hours_Per_Week
            self.App_Switch_Frequency = App_Switch_Frequency
            self.Discount_Used = Discount_Used
            self.Customer_Support_Interactions = Customer_Support_Interactions
            self.Tenure_Months = Tenure_Months
            self.Monthly_Total_Spend = Monthly_Total_Spend
            self.Income_Level = Income_Level
            self.Payment_Mode = Payment_Mode
            self.Device_Type = Device_Type

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Converts user input into pandas DataFrame
        """
        try:
            data_dict = {
                "Age": [self.Age],
                "Number_of_Subscriptions": [self.Number_of_Subscriptions],
                "Avg_Usage_Hours_Per_Week": [self.Avg_Usage_Hours_Per_Week],
                "App_Switch_Frequency": [self.App_Switch_Frequency],
                "Discount_Used": [self.Discount_Used],
                "Customer_Support_Interactions": [self.Customer_Support_Interactions],
                "Tenure_Months": [self.Tenure_Months],
                "Monthly_Total_Spend": [self.Monthly_Total_Spend],
                "Income_Level": [self.Income_Level],
                "Payment_Mode": [self.Payment_Mode],
                "Device_Type": [self.Device_Type]
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise ChurnPredictionException(e, sys)