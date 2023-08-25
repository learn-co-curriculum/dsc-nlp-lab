# Text Classification - Cumulative Lab

## Introduction

In this cumulative lab, we'll use everything we've learned so far to build a model that can classify a text document as one of many possible classes!

## Objectives

You will be able to:

- Practice cleaning and exploring a text dataset with NLTK and base Python
- Practice using scikit-learn vectorizers for text preprocessing
- Tune a modeling process through exploration and model evaluation
- Observe some techniques for feature engineering
- Interpret the result of a final ML model that classifies text data

## Your Task: Complete an End-to-End ML Process with the Newsgroups Dataset

<a title="Bundesarchiv, B 145 Bild-F077948-0006 / Engelbert Reineke / CC-BY-SA 3.0, CC BY-SA 3.0 DE &lt;https://creativecommons.org/licenses/by-sa/3.0/de/deed.en&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Bundesarchiv_B_145_Bild-F077948-0006,_Jugend-Computerschule_mit_IBM-PC.jpg"><img width="512" alt="Bundesarchiv B 145 Bild-F077948-0006, Jugend-Computerschule mit IBM-PC" src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Bundesarchiv_B_145_Bild-F077948-0006%2C_Jugend-Computerschule_mit_IBM-PC.jpg"></a>

### Business Understanding

The ***Newsgroups Dataset*** is a collection of [newsgroup](https://en.wikipedia.org/wiki/Usenet_newsgroup) posts originally collected around 1995. While the backend code implementation is fairly different, you can think of them as like the Reddit posts of 1995, where a "category" in this dataset is like a subreddit.

The task is to try to identify the category where a post was published, based on the text content of the post.

### Data Understanding

#### Data Source

Part of what you are practicing here is using the `sklearn.datasets` submodule, which you have seen before (e.g. the Iris Dataset, the Wine Dataset). You can see a full list of available dataset loaders [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets).

In this case we will be using the `fetch_20newsgroups` function ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)). An important thing to note is that because this is text data, scikit-learn actually downloads a set of documents to the computer you are using to complete this lab, rather than just loading data into memory in Python.

#### Features

Prior to preprocessing, every row in the dataset only contains one feature: a string containing the full text of the newsgroup post. We will perform preprocessing to create additional features.

#### Target

As you might have guessed based on the function name, there are 20 categories in the full dataset. Here is a list of all the possible classes:

<img src='https://curriculum-content.s3.amazonaws.com/data-science/images/classes.png'>

This full dataset is quite large. To save us from extremely long runtimes, we'll work with only a subset of the classes. For this lab, we'll work with the following five:

* `'comp.windows.x'`
* `'rec.sport.hockey'`
* `'misc.forsale'`
* `'sci.crypt'`
* `'talk.politics.misc'`

### Requirements

#### 1. Load the Data

Use pandas and `sklearn.datasets` to load the train and test data into appropriate data structures. Then get a sense of what is in this dataset by visually inspecting some samples.

#### 2. Perform Data Cleaning and Exploratory Data Analysis with `nltk`

Standardize the case of the data and use a tokenizer to convert the full posts into lists of individual words. Then compare the raw word frequency distributions of each category.

#### 3. Build and Evaluate a Baseline Model with `TfidfVectorizer` and `MultinomialNB`

Ultimately all data must be in numeric form in order to be able to fit a scikit-learn model. So we'll use a tool from `sklearn.feature_extraction.text` to convert all data into a vectorized format.

Initially we'll keep all of the default parameters for both the vectorizer and the model, in order to develop a baseline score.

#### 4. Iteratively Perform and Evaluate Preprocessing and Feature Engineering Techniques

Here you will investigate three techniques, to determine whether they should be part of our final modeling process:

1. Removing stopwords
2. Using custom tokens
3. Domain-specific feature engineering
4. Increasing `max_features`

#### 5. Evaluate a Final Model on the Test Set

Once you have chosen a final modeling process, fit it on the full training data and evaluate it on the test data. 

## 1. Load the Data

In the cell below, create the variables `newsgroups_train` and `newsgroups_test` by calling the `fetch_20newsgroups` function twice.

For the train set, specify `subset="train"`. For the test set, specify `subset="test"`.

Additionally, pass in `remove=('headers', 'footers', 'quotes')` in both function calls, in order to automatically remove some metadata that can lead to overfitting.

Recall that we are loading only five categories, out of the full 20. So, pass in `categories=categories` both times.


```python
# Replace None with appropriate code
from sklearn.datasets import fetch_20newsgroups

categories = [
    'comp.windows.x',
    'rec.sport.hockey',
    'misc.forsale',
    'sci.crypt',
    'talk.politics.misc'
]

newsgroups_train = fetch_20newsgroups(
    subset=None,
    remove=None,
    categories=None
)

newsgroups_test = fetch_20newsgroups(
    subset=None,
    remove=None,
    categories=None
)
```

Each of the returned objects is a dictionary-like `Bunch` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html)):


```python
# Run this cell without changes
type(newsgroups_train)
```

The important thing to know is that the `.data` attribute will extract the feature values, and the `.target` attribute will extract the target values. So, for example, the train features (`X_train`) are located in `newsgroups_train.data`, whereas the train targets (`y_train`) are located in `newsgroups_train.target`.

In the cell below, create `X_train`, `X_test`, `y_train`, `y_test` based on `newsgroups_train` and `newsgroups_test`.


```python
# Replace None with appropriate code
import pandas as pd
pd.set_option('max_colwidth', 400)
pd.set_option('use_mathjax', False)

# Extract values from Bunch objects
X_train = pd.DataFrame(None, columns=["text"])
X_test = pd.DataFrame(None, columns=["text"])
y_train = pd.Series(None, name="category")
y_test = pd.Series(None, name="category")
```

Double-check that your variables have the correct shape below:


```python
# Run this cell without changes

# X_train and X_test both have 1 column (text)
assert X_train.shape[1] == X_test.shape[1] and X_train.shape[1] == 1

# y_train and y_test are 1-dimensional (target value only)
assert len(y_train.shape) == len(y_test.shape) and len(y_train.shape) == 1

# X_train and y_train have the same number of rows
assert X_train.shape[0] == y_train.shape[0] and X_train.shape[0] == 2838

# X_test and y_test have the same number of rows
assert X_test.shape[0] == y_test.shape[0] and X_test.shape[0] == 1890
```

And now let's look at some basic attributes of the dataset.

#### Distribution of Target

We know that there are five categories represented. How many are there of each?


```python
# Run this cell without changes

train_target_counts = pd.DataFrame(y_train.value_counts())
train_target_counts["label"] = [newsgroups_train.target_names[val] for val in train_target_counts.index]
train_target_counts.columns = ["count", "target name"]
train_target_counts.index.name = "target value"
train_target_counts
```

So, for example, the category "comp.windows.x" has the label of `0` in our dataset, and there are 593 text samples in that category within our training data.

We also note that our target distribution looks reasonably balanced. Now let's look at the features.

#### Visually Inspecting Features

Run the cell below to view some examples of the features:


```python
# Run this cell without changes

# Sample 5 records and display full text of each
train_sample = X_train.sample(5, random_state=22)
train_sample["label"] = [y_train[val] for val in train_sample.index]
train_sample.style.set_properties(**{'text-align': 'left'})
```

In order, we have:

* An example of `comp.windows.x`, talking about "host loading considerations"
* An example of `talk.politics.misc`, talking about government and currency
* An example of `misc.forsale`, talking about a list of comics for sale
* An example of `rec.sport.hockey`, talking about hockey players and the Bruins
* An example of `sci.crypt`, talking about a microprocessor

We appear to have loaded the data correctly, so let's move on and perform some cleaning and additional exploratory analysis.

## 2. Perform Data Cleaning and Exploratory Data Analysis with `nltk`

Prior to any exploratory analysis, we'll complete two common data cleaning tasks for text data: standardizing case and tokenizing.

### Standardizing Case

In an NLP modeling process, sometimes we will want to preserve the original case of words (i.e. to treat `"It"` and `"it"` as different words, and sometimes we will want to standardize case (i.e. to treat `"It"` and `"it"` as the same word).

To figure out what we want to do, let's look at the first sample from above:


```python
# Run this cell without changes
windows_sample = train_sample.iloc[0]["text"]
windows_sample
```

Here we have two references to the company Network Computing Devices, or NCD. At the beginning, the poster refers to it as `"Ncd"`. Then later refers to `"support@ncd.com"`. It seems reasonable to assume that both of these should be treated as references to the same word instead of treating `"Ncd"` and `"ncd"` as two totally separate things. So let's standardize the case of all letters in this dataset.

The typical way to standardize case is to make everything lowercase. While it's possible to do this after tokenizing, it's easier and faster to do it first.

For a single sample, we can just use the built-in Python `.lower()` method:


```python
# Run this cell without changes
windows_sample.lower()
```

#### Standarizing Case in the Full Dataset

To access this method in pandas, you use `.str.lower()`:


```python
# Run this cell without changes

# Transform sample data to lowercase
train_sample["text"] = train_sample["text"].str.lower()
# Display full text
train_sample.style.set_properties(**{'text-align': 'left'})
```

In the cell below, perform the same operation on the full `X_train`:


```python
# Replace None with appropriate code

# Transform text in X_train to lowercase
None
```

Double-check your work by looking at an example and making sure the text is lowercase:


```python
# Run this cell without changes
X_train.iloc[100]["text"]
```

### Tokenizing

Now that the case is consistent it's time to convert each document from a single long string into a set of tokens.

Let's look more closely at the second example from our training data sample:


```python
# Run this cell without changes
politics_sample = train_sample.iloc[1]["text"]
politics_sample
```

If we split this into tokens just by using the built-in Python `.split` string method, we would have a lot of punctuation attached:


```python
# Run this cell without changes
politics_sample.split()[:10]
```

(Punctuation being attached to words is a problem because we probably want to treat `you` and `you.` as two instances of the same token, not two different tokens.)

Let's use the default token pattern that scikit-learn uses in its vectorizers. The RegEx looks like this:

```
(?u)\b\w\w+\b
```

That means:

1. `(?u)`: use full unicode string matching
2. `\b`: find a word boundary (a word boundary has length 0, and represents the location between non-word characters and word characters)
3. `\w\w+`: find 2 or more word characters (all letters, numbers, and underscores are word characters)
4. `\b`: find another word boundary

In other words, we are looking for tokens that consist of two or more consecutive word characters, which include letters, numbers, and underscores.

We'll use the `RegexpTokenizer` from NLTK to create these tokens, initially just transforming the politics sample:


```python
# Run this cell without changes

from nltk.tokenize import RegexpTokenizer

basic_token_pattern = r"(?u)\b\w\w+\b"

tokenizer = RegexpTokenizer(basic_token_pattern)
tokenizer.tokenize(politics_sample)[:10]
```

#### Tokenizing the Full Dataset

The way to tokenize all values in a column of a pandas dataframe is to use `.apply` and pass in `tokenizer.tokenize`.

For example, with the sample dataset:


```python
# Run this cell without changes

# Create new column with tokenized data
train_sample["text_tokenized"] = train_sample["text"].apply(tokenizer.tokenize)
# Display full text
train_sample.style.set_properties(**{'text-align': 'left'})
```

In the cell below, apply the same operation on `X_train`:


```python
# Replace None with appropriate code

# Create column text_tokenized on X_train
None
```

Visually inspect your work below:


```python
# Run this cell without changes
X_train.iloc[100]["text_tokenized"][:20]
```

(Note that we have removed all single-letter words, so instead of `"have", "a", "problem"`, the sample now shows just `"have", "problem"`. If we wanted to include single-letter words, we could use the token pattern `(?u)\b\w+\b` instead.)

Now that our data is cleaned up (case standardized and tokenized), we can perform some EDA.

### Exploratory Data Analysis: Frequency Distributions

Recall that a frequency distribution is a data structure that contains pieces of data as well as the count of how frequently they appear. In this case, the pieces of data we'll be looking at are tokens (words).

In the past we have built a frequency distribution "by hand" using built-in Python data structures. Here we'll use another handy tool from NLTK called `FreqDist` ([documentation here](http://www.nltk.org/api/nltk.html?highlight=freqdist#nltk.probability.FreqDist)). `FreqDist` allows us to pass in a single list of words, and it produces a dictionary-like output of those words and their frequencies.

For example, this creates a frequency distribution of the example shown above:


```python
# Run this cell without changes
from nltk import FreqDist

example_freq_dist = FreqDist(X_train.iloc[100]["text_tokenized"][:20])
example_freq_dist
```

Then can use Matplotlib to visualize the most common words:


```python
# Run this cell without changes
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def visualize_top_10(freq_dist, title):

    # Extract data for plotting
    top_10 = list(zip(*freq_dist.most_common(10)))
    tokens = top_10[0]
    counts = top_10[1]

    # Set up plot and plot data
    fig, ax = plt.subplots()
    ax.bar(tokens, counts)

    # Customize plot appearance
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", rotation=90)
    
visualize_top_10(example_freq_dist, "Top 10 Word Frequency for Example Tokens")
```

Interpreting the chart above is a bit artificial, since this sample only included 20 tokens. But essentially this is saying that the token with the highest frequency in our example is `"is"`, which occurred twice.

#### Visualizing the Frequency Distribution for the Full Dataset

Let's do that for the full `X_train`.

First, we need a list of all of the words in the `text_tokenized` column. We could do this manually by looping over the rows, but fortunately pandas has a handy method called `.explode()` ([documentation here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.explode.html#pandas.Series.explode)) that does exactly this.

Here is an example applying that to the sample dataframe:


```python
# Run this cell without changes
train_sample["text_tokenized"].explode()
```

And we can visualize the top 10 words from the sample dataframe like this:


```python
# Run this cell without changes
sample_freq_dist = FreqDist(train_sample["text_tokenized"].explode())
visualize_top_10(sample_freq_dist, "Top 10 Word Frequency for 5 Samples")
```

Note that `"00"` and `"50"` are both in the top 10 tokens, due to many prices appearing in the `misc.forsale` example.

In the cell below, complete the same process for the full `X_train`:


```python
# Replace None with appropriate code

# Create a frequency distribution for X_train
train_freq_dist = None

# Plot the top 10 tokens
None
```

Ok great, we have a general sense of the word frequencies in our dataset!

We can also subdivide this by category, to see if it makes a difference:


```python
# Run this cell without changes

# Add in labels for filtering (we won't pass them in to the model)
X_train["label"] = [y_train[val] for val in X_train.index]

def setup_five_subplots():
    """
    It's hard to make an odd number of graphs pretty with just nrows
    and ncols, so we make a custom grid. See example for more details:
    https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_multicolumn.html

    We want the graphs to look like this:
     [ ] [ ] [ ]
       [ ] [ ]

    So we make a 2x6 grid with 5 graphs arranged on it. 3 in the
    top row, 2 in the second row

      0 1 2 3 4 5
    0|[|]|[|]|[|]|
    1| |[|]|[|]| |
    """
    fig = plt.figure(figsize=(15,9))
    fig.set_tight_layout(True)
    gs = fig.add_gridspec(2, 6)
    ax1 = fig.add_subplot(gs[0, :2]) # row 0, cols 0-1
    ax2 = fig.add_subplot(gs[0, 2:4])# row 0, cols 2-3
    ax3 = fig.add_subplot(gs[0, 4:]) # row 0, cols 4-5
    ax4 = fig.add_subplot(gs[1, 1:3])# row 1, cols 1-2
    ax5 = fig.add_subplot(gs[1, 3:5])# row 1, cols 3-4
    return fig, [ax1, ax2, ax3, ax4, ax5]

def plot_distribution_of_column_by_category(column, axes, title="Word Frequency for"):
    for index, category in enumerate(newsgroups_train.target_names):
        # Calculate frequency distribution for this subset
        all_words = X_train[X_train["label"] == index][column].explode()
        freq_dist = FreqDist(all_words)
        top_10 = list(zip(*freq_dist.most_common(10)))
        tokens = top_10[0]
        counts = top_10[1]

        # Set up plot
        ax = axes[index]
        ax.bar(tokens, counts)

        # Customize plot appearance
        ax.set_title(f"{title} {category}")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="x", rotation=90)


fig, axes = setup_five_subplots()
plot_distribution_of_column_by_category("text_tokenized", axes)
fig.suptitle("Word Frequencies for All Tokens", fontsize=24);
```

If these were unlabeled, would you be able to figure out which one matched with which category?

Well, `misc.forsale` still has a number (`"00"`) as one of its top tokens, so you might be able to figure out that one, but it seems very difficult to distinguish the others; every single category has `"the"` as the most common token, and every category except for `misc.forsale` has `"to"` as the second most common token. 

After building our baseline model, we'll use this information to inform our next preprocessing steps.

## 3. Build and Evaluate a Baseline Model with `TfidfVectorizer` and `MultinomialNB`

Let's start modeling by building a model that basically only has access to the information in the plots above. So, using the default token pattern to split the full text into tokens, and using a limited vocabulary.

To give the model a little bit more information with those same features, we'll use a `TfidfVectorizer` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)) so that it counts not only the term frequency (`tf`) within a single document, it also includes the inverse document frequency (`idf`) — how rare the term is.

In the cell below, import the vectorizer, instantiate a vectorizer object, and fit it on `X_train["text"]`.


```python
# Replace None with appropriate code

# Import the relevant vectorizer class
None

# Instantiate a vectorizer with max_features=10
# (we are using the default token pattern)
tfidf = None

# Fit the vectorizer on X_train["text"] and transform it
X_train_vectorized = None

# Visually inspect the 10 most common words
pd.DataFrame.sparse.from_spmatrix(X_train_vectorized, columns=tfidf.get_feature_names_out())
```

Check the shape of your vectorized data:


```python
# Run this cell without changes

# We should still have the same number of rows
assert X_train_vectorized.shape[0] == X_train.shape[0]

# The vectorized version should have 10 columns, since we set
# max_features=10
assert X_train_vectorized.shape[1] == 10
```

Now that we have preprocessed data, fit and evaluate a multinomial Naive Bayes classifier ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)) using `cross_val_score` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)).


```python
# Replace None with appropriate code

# Import relevant class and function
None
None

# Instantiate a MultinomialNB classifier
baseline_model = None

# Evaluate the classifier on X_train_vectorized and y_train
baseline_cv = cross_val_score(None, None, None)
baseline_cv
```

How well is this model performing? Well, recall the class balance:


```python
# Run this cell without changes
y_train.value_counts(normalize=True)
```

If we guessed the plurality class every time (class `2`), we would expect about 21% accuracy. So when this model is getting 37-42% accuracy, that is a clear improvement over just guessing. But with an accuracy below 50%, we still expect the model to guess the wrong class the majority of the time. Let's see if we can improve that with more sophisticated preprocessing.

## 4. Iteratively Perform and Evaluate Preprocessing and Feature Engineering Techniques

Now that we have our baseline, the fun part begins. As you've seen throughout this section, preprocessing text data is a bit more challenging that working with more traditional data types because there's no clear-cut answer for exactly what sort of preprocessing we need to do. As we are preprocessing our text data, we need to make some decisions about things such as:

* Do we remove stop words or not?
* What should be counted as a token? Do we stem or lemmatize our text data, or leave the words as is? Do we want to include non-"words" in our tokens?
* Do we engineer other features, such as bigrams, or POS tags, or Mutual Information Scores?
* Do we use the entire vocabulary, or just limit the model to a subset of the most frequently used words? If so, how many?
* What sort of vectorization should we use in our model? Boolean Vectorization? Count Vectorization? TF-IDF? More advanced vectorization strategies such as Word2Vec?

In this lab, we will work through the first four of these.

### Removing Stopwords

Let's begin with the first question: ***do we remove stopwords or not?*** In general we assume that stopwords do not contain useful information, but that is not always the case. Let's empirically investigate the top word frequencies of each category to see whether removing stopwords helps us to distinguish between the catogories.

As-is, recall that the raw word frequency distributions of 4 out of 5 categories look very similar. They start with `the` as the word with by far the highest frequency, then there is a downward slope of other common words, starting with `to`. The `misc.forsale` category looks a little different, but it still has `the` as the top token.

If we remove stopwords, how does this change the frequency distributions for each category?

#### Stopwords List

Once again, NLTK has a useful tool for this task. You can just import a list of standard stopwords:


```python
# Run this cell without changes
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stopwords_list = stopwords.words('english')
stopwords_list[:20]
```

We can customize that list as well.

Let's say that we want to keep the word `"for"` in our final vocabulary, since it appears disproportionately often in the `misc.forsale` category. The code below removes that from the stopwords:


```python
# Run this cell without changes
print("Original list length:", len(stopwords_list))
stopwords_list.pop(stopwords_list.index("for"))
print("List length after removing 'for':", len(stopwords_list))
```

In the cell below, write a function `remove_stopwords` that takes in a list-like collection of strings (tokens) and returns only those that are not in the list of stopwords. (Use the `stopwords_list` in the global scope, so that we can later use `.apply` with this function.)


```python
# Replace None with appropriate code
def remove_stopwords(token_list):
    """
    Given a list of tokens, return a list where the tokens
    that are also present in stopwords_list have been
    removed
    """
    None
```

Test it out on one example:


```python
# Run this cell without changes
tokens_example = X_train.iloc[100]["text_tokenized"]
print("Length with stopwords:", len(tokens_example))
assert len(tokens_example) == 110

tokens_example_without_stopwords = remove_stopwords(tokens_example)
print("Length without stopwords:", len(tokens_example_without_stopwords))
assert len(tokens_example_without_stopwords) == 65
```

If that ran successfully, go ahead and apply it to the full `X_train`.


```python
# Run this cell without changes
X_train["text_without_stopwords"] = X_train["text_tokenized"].apply(remove_stopwords)
```

Now we can compare frequency distributions without stopwords:


```python
# Run this cell without changes
fig, axes = setup_five_subplots()
plot_distribution_of_column_by_category("text_without_stopwords", axes)
fig.suptitle("Word Frequencies without Stopwords", fontsize=24);
```

Ok, this seems to answer our question. The most common words differ significantly between categories now, meaning that hopefully our model will have an easier time distinguishing between them.

Let's redo our modeling process, using `stopwords_list` when instantiating the vectorizer:


```python
# Run this cell without changes

# Instantiate the vectorizer
tfidf = TfidfVectorizer(
    max_features=10,
    stop_words=stopwords_list
)

# Fit the vectorizer on X_train["text"] and transform it
X_train_vectorized = tfidf.fit_transform(X_train["text"])

# Visually inspect the vectorized data
pd.DataFrame.sparse.from_spmatrix(X_train_vectorized, columns=tfidf.get_feature_names_out())
```


```python
# Run this cell without changes

# Evaluate the classifier on X_train_vectorized and y_train
stopwords_removed_cv = cross_val_score(baseline_model, X_train_vectorized, y_train)
stopwords_removed_cv
```

How does this compare to our baseline?


```python
# Run this cell without changes
print("Baseline:         ", baseline_cv.mean())
print("Stopwords removed:", stopwords_removed_cv.mean())
```

Looks like we have a marginal improvement, but still an improvement. So, to answer ***do we remove stopwords or not:*** yes, let's remove stopwords.

### Using Custom Tokens

Our next question is ***what should be counted as a token?***

Recall that currently we are using the default token pattern, which finds words of two or more characters. What happens if we also *stem* those words, so that `swims` and `swimming` would count as the same token?

Here we have provided a custom tokenizing function:


```python
# Run this cell without changes
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language="english")

def stem_and_tokenize(document):
    tokens = tokenizer.tokenize(document)
    return [stemmer.stem(token) for token in tokens]
```

This uses `tokenizer` that we created earlier, as well as a new `stemmer` object. See an example below:


```python
# Run this cell without changes
print("Original sample:", X_train.iloc[100]["text_tokenized"][20:30])
print("Stemmed sample: ", stem_and_tokenize(X_train.iloc[100]["text"])[20:30])
```

We also need to stem our stopwords:


```python
# Run this cell without changes
stemmed_stopwords = [stemmer.stem(word) for word in stopwords_list]
```

In the cells below, repeat the modeling process from earlier. This time when instantiating the `TfidfVectorizer`, specify:

* `max_features=10` (same as previous)
* `stop_words=stemmed_stopwords` (modified)
* `tokenizer=stem_and_tokenize` (new)


```python
# Replace None with appropriate code

# Instantiate the vectorizer
tfidf = None

# Fit the vectorizer on X_train["text"] and transform it
X_train_vectorized = tfidf.fit_transform(X_train["text"])

# Visually inspect the vectorized data
pd.DataFrame.sparse.from_spmatrix(X_train_vectorized, columns=tfidf.get_feature_names_out())
```


```python
# Run this cell without changes

# Evaluate the classifier on X_train_vectorized and y_train
stemmed_cv = cross_val_score(baseline_model, X_train_vectorized, y_train)
stemmed_cv
```

How does this compare to our previous best modeling process?


```python
# Run this cell without changes
print("Stopwords removed:", stopwords_removed_cv.mean())
print("Stemmed:          ", stemmed_cv.mean())
```

Great! Another improvement, a slightly bigger one than we got when just removing stopwords. So, our best modeling process for now is one where we remove stopwords, use the default token pattern, and stem our tokens with a snowball stemmer.

### Domain-Specific Feature Engineering

The way to really get the most information out of text data is by adding features beyond just vectorizing the tokens. This code will be completed for you, and it's okay if you don't fully understand everything that is happening, but we hope it helps you brainstorm for future projects!

#### Number of Sentences

Does the number of sentences in a post differ by category? Let's investigate.

Once again, there is a tool from NLTK that helps with this task.


```python
# Run this cell without changes
from nltk.tokenize import sent_tokenize

sent_tokenize(X_train.iloc[100]["text"])
```

We can just take the length of this list to find the number of sentences:


```python
# Run this cell without changes
len(sent_tokenize(X_train.iloc[100]["text"]))
```

The following code adds a feature `num_sentences` to `X_train`:


```python
# Run this cell without changes
X_train["num_sentences"] = X_train["text"].apply(lambda x: len(sent_tokenize(x)))
```


```python
# Run this cell without changes
fig, axes = setup_five_subplots()
plot_distribution_of_column_by_category("num_sentences", axes, "Numbers of Sentences for")
fig.suptitle("Distributions of Sentence Counts by Category", fontsize=24);
```

Does this seem like a useful feature? Maybe. The distributions differ a bit, but it's hard to know if our model will pick up on this information. Let's go ahead and keep it.

#### Contains a Price

The idea here is particularly to be able to distinguish the `misc.forsale` category, but it might also help with identifying the others. Let's use RegEx to check if the text contains a price:


```python
# Run this cell without changes

# Define a price as a dollar sign followed by 1-3 numbers,
# optional commas or decimals, 1-2 numbers after the decimal
# (we're not too worried about accidentally matching malformed prices)
price_query = r'\$(?:\d{1,3}[,.]?)+(?:\\d{1,2})?'

X_train["contains_price"] = X_train["text"].str.contains(price_query)

fig, axes = setup_five_subplots()
plot_distribution_of_column_by_category("contains_price", axes, "Freqency of Posts Containing Prices for")
fig.suptitle("Distributions of Posts Containing Prices by Category", fontsize=24);
```

As we expected, the `misc.forsale` category looks pretty different from the others. More than half of those posts contain prices, whereas the overwhelming majority of posts in other categories do not contain prices. Let's include this in our final model.

#### Contains an Emoticon

This is a bit silly, but we were wondering whether different categories feature different numbers of emoticons.

Here we define an emoticon as an ASCII character representing eyes, an optional ASCII character representing a nose, and an ASCII character representing a mouth.


```python
# Run this cell without changes

emoticon_query = r'(?:[\:;X=B][-^]?[)\]3D([OP/\\|])(?:(?=\s))'

X_train["contains_emoticon"] = X_train["text"].str.contains(emoticon_query)

fig, axes = setup_five_subplots()
plot_distribution_of_column_by_category("contains_emoticon", axes, "Freqency of Posts Containing Emoticons for")
fig.suptitle("Distributions of Posts Containing Emoticons by Category", fontsize=24);
```

Well, that was a lot less definitive. Emoticons are fairly rare across categories. But, there are some small differences so let's go ahead and keep it.

#### Modeling with Vectorized Features + Engineered Features 

Let's combine our best vectorizer with these new features:


```python
# Run this cell without changes

# Instantiate the vectorizer
tfidf = TfidfVectorizer(
    max_features=10,
    stop_words=stemmed_stopwords,
    tokenizer=stem_and_tokenize
)

# Fit the vectorizer on X_train["text"] and transform it
X_train_vectorized = tfidf.fit_transform(X_train["text"])

# Create a full df of vectorized + engineered features
X_train_vectorized_df = pd.DataFrame(X_train_vectorized.toarray(), columns=tfidf.get_feature_names_out())
preprocessed_X_train = pd.concat([
    X_train_vectorized_df, X_train[["num_sentences", "contains_price", "contains_emoticon"]]
], axis=1)
preprocessed_X_train
```


```python
# Run this cell without changes
preprocessed_cv = cross_val_score(baseline_model, preprocessed_X_train, y_train)
preprocessed_cv
```


```python
# Run this cell without changes
print("Stemmed:           ", stemmed_cv.mean())
print("Fully preprocessed:", preprocessed_cv.mean())
```

Ok, another small improvement! We're still a bit below 50% accuracy, but we're getting improvements every time.

### Increasing `max_features`

Right now we are only allowing the model to look at the tf-idf of the top 10 most frequent tokens. If we allow it to look at all possible tokens, that could lead to high dimensionality issues (especially if we have more rows than columns), but there is a lot of room between 10 and `len(X_train)` features:


```python
# Run this cell without changes
len(X_train)
```

(In other words, setting `max_features` to 2838 would mean an equal number of rows and columns, something that can cause problems for many model algorithms.)

Let's try increasing `max_features` from 10 to 200:


```python
# Replace None with appropriate code

# Instantiate the vectorizer
tfidf = TfidfVectorizer(
    max_features=None,
    stop_words=stemmed_stopwords,
    tokenizer=stem_and_tokenize
)

# Fit the vectorizer on X_train["text"] and transform it
X_train_vectorized = tfidf.fit_transform(X_train["text"])

# Create a full df of vectorized + engineered features
X_train_vectorized_df = pd.DataFrame(X_train_vectorized.toarray(), columns=tfidf.get_feature_names_out())
final_X_train = pd.concat([
    X_train_vectorized_df, X_train[["num_sentences", "contains_price", "contains_emoticon"]]
], axis=1)
final_X_train
```


```python
# Run this cell without changes

final_cv = cross_val_score(baseline_model, final_X_train, y_train)
final_cv
```

Nice! Our model was able to learn a lot more with these added features. Let's say this is our final modeling process and move on to a final evaluation.

## 5. Evaluate a Final Model on the Test Set

Instantiate the model, fit it on the full training set and check the score:


```python
# Run this cell without changes
final_model = MultinomialNB()

final_model.fit(final_X_train, y_train)
final_model.score(final_X_train, y_train)
```

Create a vectorized version of `X_test`'s text:


```python
# Run this cell without changes

# Note that we just transform, don't fit_transform
X_test_vectorized = tfidf.transform(X_test["text"])
```

Feature engineering for `X_test`:


```python
# Run this cell without changes
X_test["num_sentences"] = X_test["text"].apply(lambda x: len(sent_tokenize(x)))
X_test["contains_price"] = X_test["text"].str.contains(price_query)
X_test["contains_emoticon"] = X_test["text"].str.contains(emoticon_query)
```

Putting it all together:


```python
# Run this cell without changes
X_test_vectorized_df = pd.DataFrame(X_test_vectorized.toarray(), columns=tfidf.get_feature_names_out())
final_X_test = pd.concat([
    X_test_vectorized_df, X_test[["num_sentences", "contains_price", "contains_emoticon"]]
], axis=1)
final_X_test
```

Scoring on the test set:


```python
# Run this cell without changes
final_model.score(final_X_test, y_test)
```

Plotting a confusion matrix:


```python
# Run this cell without changes
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cnf_matrix = confusion_matrix(y_test, final_model.predict(final_X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=final_model.classes_)
disp.plot(cmap=plt.cm.Blues)
```

Recall that these are the names associated with the labels:


```python
# Run this cell without changes
target_values_and_names = train_target_counts.drop("count", axis=1)
target_values_and_names
```

### Interpreting Results

Interpret the results seen above. How well did the model do? How does it compare to random guessing? What can you say about the cases that the model was most likely to mislabel? If this were a project and you were describing next steps, what might those be?


```python
# Replace None with appropriate text
"""
None
"""
```

## Summary

In this lab, we used our NLP skills to clean, preprocess, explore, and fit models to text data for classification. This wasn't easy — great job!!
