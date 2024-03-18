import pickle

with open('transform.pkl', 'rb') as file:
  data = pickle.load(file)

print(data)
