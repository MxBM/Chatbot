import sqlite3
import pandas as pd

timeframes = ["2017-11"]

for timeframe in timeframes:
    connection = sqlite3.connect("{}.db".format(timeframe))  # datbase file
    c = connection.cursor()
    limit = 5000  # Limit for the pull
    last_unix = 0  # Buffer in each pull
    cur_length = limit
    counter = 0
    test_done = False

    while (
        cur_length == limit
    ):  # Pull exhausted the limit as zero rows of data were left
        df = pd.read_sql(
            "SELECT * FROM parent_reply WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format(
                last_unix, limit
            ),
            connection,
        )
        last_unix = df.tail(1)["unix"].values[0]
        cur_length = len(df)
        if not test_done:
            with open("test.from", "a", encoding="utf8") as f:
                for content in df["parent"].values:
                    f.write(content + "\n")
            with open("test.to", "a", encoding="utf8") as f:
                for content in df["comment"].values:
                    f.write(content + "\n")
            # TODO: Update cur_length and limit here
            test_done = True
        else:  # Assuming test is complete
            with open(
                "train.from", "a", encoding="utf8"
            ) as f:  # TODO: Turn this into a function
                for content in df["parent"].values:
                    f.write(content + "\n")
            with open("train.to", "a", encoding="utf8") as f:
                for content in df["comment"].values:
                    f.write(content + "\n")
        counter += 1

        if counter % 20 == 0:
            print(counter * limit, "rows completed so far")
