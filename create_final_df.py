import pandas as pd
import os.path

READING_PATH = './data/daily_report_per_states/'

def get_date_column_as_df():
    df = pd.read_csv(READING_PATH+"Alabama.csv")
    return df['Date']

class CreateDataframe(object):
     
    def get_final_df(self, type):
        df = get_date_column_as_df()
        for root,dirs,files in os.walk(READING_PATH):
            for file in files:
                if file.endswith(".csv") and file != "states.csv":
                    state_name = file[:-4]
                    state_df = pd.read_csv(READING_PATH+file)
                    state_df = state_df.rename(columns={type: state_name})
                    df = pd.concat([df, state_df[state_name]], axis=1)
        df = df.assign(Days=[1 + i for i in range(len(df))])[['Days'] + df.columns.tolist()]  
        return df        
             