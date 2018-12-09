library(magrittr) #Allows use of pipe functionality 
library(pdftools) # Reads pdf documents into text strings
library(tm) # Text cleaning for large corpa similar to tidytext tokenizing
library(quanteda) # Text cleaning for large corpa similar to tidytext tokenizing
library(tidytext) # For analysis of text in a tidy manner including sentiment data
library(textstem) # For stemming and lemmatizing text

library(lsa) # For latent semantic analysis
library(stm) # For structural topic modeling
#library(uwot) #For umap dimensionality reduction 1

library(knitr)
library(rPref) # For pareto frontier 
library(ggpubr) # 
library(ggrepel)
library(grid)
library(tidyverse)


setwd("/Users/Rachel/Documents/ISyE 859/")
data_path <- "Inverviews3"   # path to files
files <- dir(data_path, pattern = "*.pdf") # files to read

docs = data_frame(document = files) %>%         # creates a data frame with file names
  mutate(text = map(document,              
                    ~ pdf_text(file.path(data_path, .))) # reads and converts pdf files to text
  )  
# Creates clean name and separates into file number and name
docs$document = gsub('.pdf', '', docs$document)  # removes extension from file name

## Tokenize (extract letters, words, biagram, sentences, paragraphs) into a tidy format
hcc_stopwords <- tibble(word = c("interviewer", "respondent", "ninterviewer", "nrespondent", "nand", "nthat", "nbecause", "nshe"))
stopws <- tibble(word1 = c("i", "me", "my", "myself", "we", "our", "ours","ourselves",
"you", "your", "yours", "yourself", "yourselves", "he", "him",
"his", "himself", "she", "her", "hers", "herself", "it",
"its", "itself", "they", "them", "their", "theirs", "themselves",
"what", "which", "who", "whom", "this", "that", "these", "those",
"am", "is", "are", "was", "were", "be", "been", "being", "have",
"has", "had", "having", "do", "does", "did", "doing", "would",
"should", "could", "ought", "i’m", "i'm", "you’re", "you're", "he’s", "he's", "she’s", "she's", "it’s", "it's",
"we’re", "we're", "they’re", "they're", "i’ve", "i've", "you’ve", "you've", "we’ve", "we've", "they’ve", "they've", "i’d", "i'd",
"you’d", "you'd", "he’d", "he'd", "she’d", "she'd", "we’d", "we'd", "they’d","they'd", "i’ll", "i'll", "you’ll", "you'll", "he’ll", "he'll",
"she’ll", "she'll", "we’ll", "we'll", "they’ll", "they'll", "isn’t", "isn't", "aren’t", "aren't", "wasn’t", "wasn't", "weren’t", "weren't", "hasn’t", "hasn't",
"haven’t", "haven't", "hadn’t", "hadn't", "doesn’t", "doesn't", "don’t", "don't", "didn’t", "didn't", "won’t", "won't", "wouldn’t", "wouldn't",
"shan’t","shan't", "shouldn’t", "shouldn't", "can’t", "can't", "cannot", "couldn’t", "couldn't", "mustn’t", "mustn't",
"let’s", "let's", "that’s", "that's", "who’s", "who's", "what’s", "what's", "here’s", "here's", "there’s", "there's", "when’s", "when's", "where’s",
"where's","why’s", "why's", "how’s", "how's", "a", "an", "the", "and", "but", "if", "or", "because",
"as", "until", "while", "of", "at", "by", "for", "with", "about",
"against", "between", "into", "through", "during", "before", "after",
"above", "below", "to", "from", "up", "down", "in", "out", "on",
"of", "over", "under", "again", "further", "then", "once", "here",
"there", "when", "where", "why", "how", "all", "any", "both",
"each", "few", "more", "most", "other", "some", "such", "no",
"nor", "not", "only", "own", "same", "so", "than", "too", "very", "will", "that’s"))


## Function to clean and tokenize
tokenize_docs <- function(docs) {
  # Takes tidy
  # Remove numbers
  #docs$text = gsub('[0-9]+', '', docs$text) # Removes words that include numbers
  docs$text = gsub('[0-9]', '', docs$text) # Removes numbers
  # Tokenize based on word as token and remove punctuation and convert to lower case
  text.df = docs %>%
    unnest_tokens(term, text, token = "words",
                  to_lower = TRUE, strip_punct = TRUE)

# Remove stopwords
text.df = text.df %>% anti_join(stopws, by = c("term" = "word1"))

# Remove one-letter words
text.df = text.df %>% filter(str_length(term)>2)

# Remove very long words--spurious words created by pdf reader
#text.df = text.df %>% filter(str_length(term)<20)

#Remove document - specific words
text.df = text.df %>% anti_join(hcc_stopwords, by = c("term" = "word"))
return(text.df)
}
text.df = tokenize_docs(docs)

## Calculate term frequency and add tf_idf variables
tfidf.text.df = text.df %>% count(document, term) %>% 
  bind_tf_idf(term, document, n)

## Filter infrequent words
tfidf.text.df = tfidf.text.df %>% filter(n>8)

## Filter indiscriminant words--very low tf_idf words
tfidf.text.df = tfidf.text.df %>% filter(tf_idf>.000001)

## Plot most discriminating terms
top10.df = tfidf.text.df %>% group_by(document) %>% top_n(10, tf_idf) 

ggplot(top10.df, aes(factor(tf_idf), tf_idf)) +
  geom_col() +
  coord_flip() +
  facet_wrap(.~document, scales = "free")+
  scale_x_discrete( # Needs to order the term frequency
    breaks = top10.df$tf_idf, 
    labels = top10.df$term) 

## Plot number of words in processed document
length.df = tfidf.text.df %>% group_by(document) %>% summarize(words = sum(n))

ggplot(length.df, aes(words)) +
  geom_step(stat = "ecdf")

## Convert tidytext to spars metrix for stm analysis
text.sparse = tfidf.text.df %>% cast_sparse(document, term, n)

## Fit structural topic model for a range of topics
multi_stm.fit = searchK(text.sparse, 
                        K= c(4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26), 
                        M = 10, # number of top words for exclusivity calculation
                        init.type = "Spectral",
                        N=6, proportion = 0.25, # Held out documents for likelihood calculation
                        heldout.seed = 888, cores = 4)

## Plot topic metrics
# Higher heldout likelihodd 
# Lower residual dispersion a value  greater than suggests a need for more topics
# Higher semantic coherence and exclusivity
# Higher lower bound of the marginal likelihood (evidence)
plot(multi_stm.fit)


## Plot pareto curve of exclusivity and coherence 
# Coherence (high probablity terms of a topic occur together in documents)
# Exclusivity (high probablity terms of a topic are not high probability tersm in other topics)
# Mimno, D., Wallach, H. M., Talley, E., & Leenders, M. (2011). Optimizing Semantic Coherence in Topic Models, (2), 262–272.

multifit.results = multi_stm.fit$results

sky <- psel(multifit.results, high(semcoh) * high(exclus))
select.plot = ggplot(multifit.results, aes(semcoh, exclus))+
  geom_point()+
  geom_text_repel(aes(label = K)) +
  geom_step(data = sky, direction = "vh") +
  labs(y = "Exclusivity", x = " Coherence")
select.plot

# Fit topic model
topic_model <- stm(text.sparse, K = 13, verbose = FALSE, init.type = "Spectral")

# Extract word-topic combinations
td_beta <- tidy(topic_model) 

# Plot defining terms for topics
td_beta =
  td_beta %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup()

ggplot(td_beta, aes(factor(beta), beta)) +
  geom_col() +
  facet_wrap(~ topic, scales = "free") +
  scale_x_discrete( # Needs to order the term frequency
    breaks = td_beta$beta, 
    labels = td_beta$term) +
  coord_flip()
