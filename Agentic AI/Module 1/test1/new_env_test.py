import pandas as pd
import numpy as np

def main():
    print("NumPy array example:")
    arr = np.array([1, 2, 3, 4, 5])
    print(arr)

    print("\nPandas DataFrame example:")
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]
    }
    df = pd.DataFrame(data)
    print(df)

if __name__ == "__main__":
    main()