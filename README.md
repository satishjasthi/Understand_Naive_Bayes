# Lets learn Naive Bayes

Naive Bayes is one of the simplest machine learning models which uses Bayes theorem at its core. To understand Naive Bayes one should first understand Bayes theorem which in turn uses conditional probability.

So let's start understanding conditional probability(CP) first. Mathematically CP is defined as 

```
            P(A,B)
P(A|B) = --------------
            P(B)
```   
ie probability of A given B is probability of A and B by probability of B. But it seems a bit difficult to understand, let's try to understand with a simple example.

Let's say we have a special cheese burst pizza which has four slices and each slices has a different toppings as shown below
![alt pizza image](/images/pizza.png)

where 
- s1 = corn as toppings
- s2 = olives as toppings
- s3 = onions as toppings
- s4 = mushrooms as toppings

Now, now what portion of pizza corresponds to a slice which has corn as toppings and pizza type as cheese burst pizza. It is obvious, one fourth of pizza right. So if we redefine the above question in terms of probability it will be something like this

What is the probability of picking a slice that has corn as toppings given that it is a cheese burst pizza?

```
P(slice_toppings = corn | pizza_type = cheese_burst)
```
So we have got the answer for left hand side ie 

```
                                                        1
P(slice_toppings = corn | pizza_type = cheese_burst) = ---
                                                        4
```

However, 

```
                                                          P(slice_toppings = corn , pizza_type = cheese_burst)
P(slice_toppings = corn | pizza_type = cheese_burst) = -------------------------------------------------------------
                                                                        P(pizza_type = cheese_burst)
```

That means,

```

 1        P(slice_toppings = corn , pizza_type = cheese_burst)
--- = -------------------------------------------------------------
 4                     P(pizza_type = cheese_burst)
```
Funny thing is that we were able to get the value of 
```
P(slice toppings = corn | pizza type = cheese burst)
```
without actually using formula. Let's how can we validate above equation 

We know that 

```
P(pizza_type = cheese_burst) = 1
```
because there is only one cheese crust pizza

and now 

```
                                                        1
P(slice_toppings = corn , pizza_type = cheese_burst) = ----
                                                        4

```
because 

```
        Number of favourable events for A           Number of slices with corn toppings       1
P(A) = ------------------------------------    =  --------------------------------------- = ---------
           Total number of events                   Total number of slices with toppings      4
```

Now that we are clear with CP let's understand how Bayes theorem(BT) works. Mathematically BT is defined as 

```
          P(E|H) * P(H)
P(H|E) = -----------------
              P(E)
                

``` 
where
```
- P(H) is the probability of hypothesis H being true. This is known as the prior probability.
- P(E) is the probability of the evidence(regardless of the hypothesis).
- P(E|H) is the probability of the evidence given that hypothesis is true.
- P(H|E) is the probability of the hypothesis given that the evidence is there.
```

Wowo that's pretty hard to understand...
Let's take a example and try to simplify the concept,

Consider a Lab is performing a test of disease say “D” with two results “Positive” & “Negative.” They guarantee that their test result is 99% accurate: if you have the disease, they will give test positive 99% of the time. If you don’t have the disease, they will test negative 99% of the time. If 3% of all the people have this disease and test gives “positive” result, what is the probability that you actually have the disease?

For solving the above problem, we will have to use conditional probability.
Probability of people suffering from Disease D, P(D) = 0.03 = 3%
Probability that test gives “positive” result and patient have the disease, P(Pos | D) = 0.99 =99%

Probability of people not suffering from Disease D, P(~D) = 0.97 = 97%
Probability that test gives “positive” result and patient does have the disease, P(Pos | ~D) = 0.01 =1%

For calculating the probability that the patient actually have the disease i.e, 
```
P( D | Pos) 
```
we will use Bayes theorem:

```
                P(Pos | D) * P(D)
P(D | Pos) = -------------------------
                   P(Pos)
```
We have all the values of numerator but we need to calculate P(Pos):

```
P(Pos) = P(D, pos) + P( ~D, pos)
```
ie Probability of being positive can be either when the test says it is positive and patient has disease or when the test says it is positive but patient has no disease

```
= P(pos|D)*P(D) + P(pos|~D)*P(~D)
= 0.99 * 0.03 + 0.01 * 0.97
= 0.0297 + 0.0097
= 0.0394
```

Let’s calculate,

```
P( D | Pos) = (P(Pos | D) * P(D)) / P(Pos)
= (0.99 * 0.03) / 0.0394
= 0.753807107
```

So, Approximately 75% chances are there that the patient is actually suffering from disease when the test confirms disease.

Now that we are clear about Bayes theorem let's understand Naive Bayes

# Naive Bayes Classifier

Naive Bayes is a kind of classifier which uses the Bayes Theorem. It predicts membership probabilities for each class such as the probability that given record or data point belongs to a particular class.  The class with the highest probability is considered as the most likely class. This is also known as Maximum A Posteriori (MAP).
The MAP for a hypothesis is:

```
MAP(H)
= max( P(H|E) )
=  max( (P(E|H)*P(H))/P(E))
= max(P(E|H)*P(H))
```

P(E) is evidence probability, and it is used to normalize the result. It remains same so, removing it won’t affect.

Naive Bayes classifier assumes that all the features are unrelated to each other. Presence or absence of a feature does not influence the presence or absence of any other feature. We can use Wikipedia example for explaining the logic i.e.,

```
A fruit may be considered to be an apple if it is red, round, and about 4″ in diameter.  Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier considers all of these properties to independently contribute to the probability that this fruit is an apple.
```

In real datasets, we test a hypothesis given multiple evidence(feature). So, calculations become complicated. To simplify the work, the feature independence approach is used to ‘uncouple’ multiple evidence and treat each as an independent one.

```
P(H|Multiple Evidences) =  P(E1| H)* P(E2|H) ……*P(En|H) * P(H) / P(Multiple Evidences)
```
# Example of Naive Bayes Classifier

Let's try to understand whatever theory we have learnt about Naive Bayes by implementing it on example.

Let’s consider a training dataset with 1500 records and 3 classes. We presume that there are no missing values in our data. We have
We have 3 classes associated with Animal Types:

- Parrot,
- Dog,
- Fish.

The Predictor features set consists of 4 features as

- Swim
- Wings
- Green Color
- Dangerous Teeth.

Green Color, Dangerous Teeth.  All the features are categorical variables with either of the 2 values: T(True) or F( False).

| Swim          | Wings         | Green Color  | Dangerous Teeth | Animal Type |
|:-------------:|:-------------:|:------------:|:---------------:|:-----------:|
| 50            | 500/500       |  400/500     |  0              | Parrot      |
| 450/500       | 0             |  0           |  500/500        | Dog         |
| 500/500       | 0             |  100/500     |  50/500         | Fish        |


The above table shows a frequency table of our data. In our training data:

- Parrots have 50(10%) value for Swim, i.e., 10% parrot can swim according to our data, 500 out of 500(100%) parrots have wings, 400 out of 500(80%) parrots are Green and 0(0%) parrots have Dangerous Teeth.
- Classes with Animal type Dogs shows that 450 out of 500(90%) can swim, 0(0%) dogs have wings, 0(0%) dogs are of Green color and 500 out of 500(100%) dogs have Dangerous Teeth.
- Classes with Animal type Fishes shows that 500 out of 500(100%) can swim, 0(0%) fishes have wings, 100(20%) fishes are of Green color and 50 out of 500(10%) dogs have Dangerous Teeth.

Now, it’s time to work on predict classes using the Naive Bayes model. We have taken 2 records that have values in their feature set, but the target variable needs to predicted.

|               | Swim          | Wings        | Green Color     |Dangerous Teeth | 
|:-------------:|:-------------:|:------------:|:---------------:|:--------------:|
| 1.            | True          |  False       |  True           | False          |
| 2.            | True          |  False       |  True           | True           |

We have to predict animal type using the feature values. We have to predict whether the animal is a Dog, a Parrot or a Fish

We will use the Naive Bayes approach
```
P(H|Multiple Evidences) =  P(E1| H)* P(E2|H) ……*P(En|H) * P(H) / P(Multiple Evidences)
```
Let’s consider the first record.
The Evidence here is Swim & Green. The Hypothesis can be an animal type to be Dog, Parrot, Fish.
For Hypothesis testing for the animal to be a Dog:

```
P(Dog | Swim, Green) = P(Swim|Dog) * P(Green|Dog) * P(Dog) / P(Swim, Green)
=  0.9 * 0 * 0.333 / P(Swim, Green)
= 0
```

For Hypothesis testing for the animal to be a Parrot:
```
P(Parrot| Swim, Green) = P(Swim|Parrot) * P(Green|Parrot) * P(Parrot) / P(Swim, Green)
=  0.1 * 0.80 * 0.333 / P(Swim, Green)
= 0.0264/ P(Swim, Green)
```

For Hypothesis testing for the animal to be a Fish:
```
P(Fish| Swim, Green) = P(Swim|Fish) * P(Green|Fish) * P(Fish) / P(Swim, Green)
=  1 * 0.2 * 0.333 / P(Swim, Green)
= 0.0666/ P(Swim, Green)
```
The denominator of all the above calculations is same i.e, P(Swim, Green). The value of P(Fish| Swim, Green) is greater that P(Parrot| Swim, Green).

Using Naive Bayes, we can predict that the class of this record is Fish.

Let’s consider the second record.
The Evidence here is Swim, Green & Teeth. The Hypothesis can be an animal type to be Dog, Parrot, Fish.
For Hypothesis testing for the animal to be a Dog:
```
P(Dog | Swim, Green, Teeth) = P(Swim|Dog) * P(Green|Dog) * P(Teeth|Dog) * P(Dog) / P(Swim, Green, Teeth)
=  0.9 * 0 * 1 * 0.333 / P(Swim, Green, Teeth)
= 0
```
For Hypothesis testing for the animal to be a Parrot:
```
P(Parrot| Swim, Green, Teeth) = P(Swim|Parrot) * P(Green|Parrot)* P(Teeth|Parrot) * P(Parrot) / P(Swim, Green, Teeth)
=  0.1 * 0.80 *  0 *0.333 / P(Swim, Green, Teeth)
= 0
```
For Hypothesis testing for the animal to be a Fish:
```
P(Fish|Swim, Green, Teeth) = P(Swim|Fish) * P(Green|Fish) * P(Teeth|Fish) *P(Fish) / P(Swim, Green, Teeth)
=  1 * 0.2 * 0.1 * 0.333 / P(Swim, Green, Teeth)
= 0.00666 / P(Swim, Green, Teeth)
```
The denominator of all the above calculations is same i.e, P(Swim, Green, Teeth). The value of P(Fish| Swim, Green, Teeth) is the only positive value greater than 0. Using Naive Bayes, we can predict that the class of this record is Fish.

As the calculated value of probabilities is very less. To normalize these values, we need to use denominators.

# Pros and Cons of Naive Bayes classifier

Pros

- Naive Bayes Algorithm is a fast, highly scalable algorithm.
- Naive Bayes can be use for Binary and Multiclass classification. It provides different types of Naive Bayes Algorithms like GaussianNB, MultinomialNB, BernoulliNB.
- It is a simple algorithm that depends on doing a bunch of counts.
- Great choice for Text Classification problems. It’s a popular choice for spam email classification.
- It can be easily train on small dataset

Cons

- It considers all the features to be unrelated, so it cannot learn the relationship between features. E.g., Let’s say Remo is going to a part. While cloth selection for the party, Remo is looking at his cupboard. Remo likes to wear a white color shirt. In Jeans, he likes to wear a brown Jeans, But Remo doesn’t like wearing a white shirt with Brown Jeans. Naive Bayes can learn individual features importance but can’t determine the relationship among features.

references:
[Naive Bayes wiki](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
[Naive Bayes by Rahul Saxena](http://dataaspirant.com/2017/02/06/naive-bayes-classifier-machine-learning/)

