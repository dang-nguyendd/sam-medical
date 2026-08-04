import pickle

# Open the file in read-binary mode
with open('./model_pth/YOLOSAM_v1_run1_CVC-ColonDB/training_history.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)