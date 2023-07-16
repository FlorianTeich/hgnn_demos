def sin_transformer(period):
    from sklearn.preprocessing import FunctionTransformer
    import numpy as np
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    from sklearn.preprocessing import FunctionTransformer
    import numpy as np
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def proc_raw(data):
    return data.values


def proc_objects_one_hot(data):
    import pandas as pd
    return pd.get_dummies(data).values


def proc_objects_string(data):
    from sentence_transformers import SentenceTransformer
    import torch

    model_name='all-MiniLM-L6-v2'
    model_string_encoder = SentenceTransformer(model_name)

    @torch.no_grad()
    def encode_strings(df):
        x = model_string_encoder.encode(df.values, show_progress_bar=True)
        return x
    return encode_strings(data.fillna(''))


def postproc(data):
    import numpy as np

    for i, d in enumerate(data):
        if len(d.shape) == 1:
            data[i] = np.expand_dims(d, 1)
    return np.hstack(data)


def proc_datetime(data):
    import numpy as np
    feature_tmp_year = data.dt.year
    feature_tmp_day = data.dt.day
    # Cyclical features:
    feature_tmp_month = data.dt.month
    feature_tmp_hour = data.dt.hour
    feature_tmp_minute = data.dt.minute
    feature_tmp_second = data.dt.second
    feature_tmp_day_of_week = data.dt.day_of_week

    feature_tmp_month_x = sin_transformer(12).fit_transform(feature_tmp_month)
    feature_tmp_month_y = cos_transformer(12).fit_transform(feature_tmp_month)

    feature_tmp_hour_x = sin_transformer(60).fit_transform(feature_tmp_hour)
    feature_tmp_hour_y = cos_transformer(60).fit_transform(feature_tmp_hour)

    feature_tmp_minute_x = sin_transformer(60).fit_transform(feature_tmp_minute)
    feature_tmp_minute_y = cos_transformer(60).fit_transform(feature_tmp_minute)

    feature_tmp_second_x = sin_transformer(60).fit_transform(feature_tmp_second)
    feature_tmp_second_y = cos_transformer(60).fit_transform(feature_tmp_second)

    feature_tmp_day_of_week_x = sin_transformer(7).fit_transform(feature_tmp_day_of_week)
    feature_tmp_day_of_week_y = cos_transformer(7).fit_transform(feature_tmp_day_of_week)

    feature_tmp = [feature_tmp_year, 
                    feature_tmp_month_x,
                    feature_tmp_month_y,
                    feature_tmp_day,
                    feature_tmp_hour_x,
                    feature_tmp_hour_y,
                    feature_tmp_minute_x,
                    feature_tmp_minute_y,
                    feature_tmp_second_x,
                    feature_tmp_second_y,
                    feature_tmp_day_of_week_x,
                    feature_tmp_day_of_week_y,
                    ]
    return np.vstack(feature_tmp).T


def edges_trans():
    data = pd.read_parquet("./data/edges_loan_account.parquet")
    return np.flip(data[["loan_id", "account_id"]].values.T)
