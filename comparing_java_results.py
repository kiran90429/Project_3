import pandas as pd

data_java = pd.read_csv(r'C:\\Users\\kiran.kandula\\PycharmProjects\\JESUS\\Project_3-PnG-\\history2\\results2.csv')

data_frame_java = pd.DataFrame(data_java.values,columns=['java'])