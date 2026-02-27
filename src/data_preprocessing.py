import numpy as np
import pandas as pd


train = pd.read_csv('../data/raw/train.csv', dtype={'StateHoliday': str})
store = pd.read_csv('../data/raw/store.csv')

df = pd.merge(train, store, on='Store', how='left')
df.to_csv('../data/merge.csv', index=False)

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.day_of_week  # 0 = Monday, 6 = Sunday

df = df.drop('Date', axis=1)


# فاصله → میانه (مقاوم‌تر به داده‌های پرت)
df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())

# اگر رقابتی وجود ندارد → علامت‌گذاری واضح
df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)

# Promo2 فعال نیست
df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
df['PromoInterval'] = df['PromoInterval'].fillna('None')

df = df[df['Open'] == 1]
# دیگه به Open نیازی نداریم
df = df.drop(columns=['Open'], errors='ignore')

# -------------------------
# ۳️⃣ One-Hot Encoding تمام ویژگی‌های دسته‌ای
# -------------------------

# ۱. تمیز کردن StateHoliday (خیلی مهم - گاهی 0 و '0' قاطی می‌شن)
df['StateHoliday'] = df['StateHoliday'].astype(str).replace('0', 'None')

# اضافه کردن یک ستون باینری ساده (خیلی به مدل کمک می‌کنه)
df['IsStateHoliday'] = (df['StateHoliday'] != 'None').astype(int)

# ۲. لیست ستون‌هایی که می‌خواهیم One-Hot کنیم
categorical_columns = [
    'StateHoliday',
    'StoreType',
    'Assortment',
    'PromoInterval'
]

# ۳. انجام One-Hot Encoding با pandas (ساده و قابل کنترل)
df = pd.get_dummies(
    df,
    columns=categorical_columns,
    prefix=categorical_columns,  # اسم ستون‌ها خوانا بمونه
    prefix_sep='_',
    drop_first=False,  # همه دسته‌ها رو نگه می‌داریم
    dtype=int  # 0 و 1 به جای True/False
)

df.to_csv('../data/preprocessed/preprocessed.csv', index=False)
