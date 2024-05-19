
def create_features(df):
    df = df.copy()
    df['source'] = df['source_prefix'] + df['source_postfix'].astype(str)
    df['destination'] = df['dest_prefix'] + df['dest_postfix'].astype(str)
    df['source_destination'] = df['source'] + '_' + df['destination']
    return df

def drop_columns(df):
    return df.drop(columns=['source_prefix', 'source_postfix', 'dest_prefix', 'dest_postfix'])