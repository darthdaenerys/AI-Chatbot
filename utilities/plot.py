import matplotlib.pyplot as plt
import seaborn as sns

def show_qna_tokens(df):
    plt.style.use('fivethirtyeight')
    fig,ax=plt.subplots(nrows=3,ncols=1,figsize=(20,18))
    sns.set_palette('Set2')
    sns.countplot(x=df['question tokens'],data=df,ax=ax[0])
    sns.countplot(x=df['answer tokens'],data=df,ax=ax[1])
    sns.boxplot(x=df['question tokens'],y=df['answer tokens'],ax=ax[2])
    plt.show()
