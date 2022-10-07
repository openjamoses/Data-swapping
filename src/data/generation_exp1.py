#from splitters import split_by_percent
import random

train_percentage = 2.0/3.0

def load_data():
  N = 6000
  headers = ["Feature A (i)", "Feature B (2i)", "Feature C (-i)",
             "Constant Feature", "Random Feature", "Outcome"]

  data = [[i, 2*i, -i, 1, random.random(), "A"] for i in range(0,int(N/2))] + \
          [[i, 2*i, -i, 1, random.random(), "B"] for i in range(int(N/2),N)]

  #train, test = split_by_percent(data, train_percentage)

  return headers, data #train, test

if __name__ == '__main__':
    headers, data = load_data()
    print(data)