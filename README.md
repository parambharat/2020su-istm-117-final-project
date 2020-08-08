# Discriminative segmentation of long speech recognition transcripts.

This work is developed the for the final project of Text Analytics and NLP (ISTM-117).
The project aims to segment Automatic Speech Recognition(ASR) transcripts into cohesive topics using transformers and bi-directional LSTMs

## Installation
The project makes use of the virtual environment `pipenv` for development and deployment.
To install the required dependencies run the following command in the terminal.

```bash
make install
``` 

## Deployment
To deploy the model as an api run the following command:
```bash
make deploy
```

Once the api is up and running, use an application like [Postman](https://www.postman.com/) to POST a request with the following structure
```json
Request:
{
  "text": "very .... long .. transcript"
}
``` 

use the following curl request for the above example
```bash
curl --location --request POST 'bebop:8000/segment_text' \
--header 'Content-Type: application/json' \
--data-raw '{"text": "euclidean division is the mathematical formulation of the outcome of the usual process of division of integers it asserts that given two integers the dividend and the divisor such that there are unique integers the quotient and the remainder such that bq and where denotes the absolute value of of integers integers are not closed under division apart from division by zero being undefined the quotient is not an integer unless the dividend is an integer multiple of the divisor for example cannot be divided by to give an integer such case uses one of five approaches say that cannot be divided by division becomes partial function give an approximate answer as real number this is the approach usually taken in numerical computation give the answer as fraction representing rational number so the result of the division of by is displaystyle tfrac tfrac or as mixed number so displaystyle tfrac tfrac displaystyle tfrac tfrac usually the resulting fraction should be simplified the result of the division of by is also displaystyle tfrac tfrac this simplification may be done by factoring out the greatest common divisor give the answer as an integer quotient and remainder so displaystyle tfrac mbox remainder tfrac mbox remainder to make the distinction with the previous case this division with two integers as result is sometimes called euclidean division because it is the basis of the euclidean algorithm give the integer quotient as the answer so displaystyle tfrac tfrac this is sometimes called integer division dividing integers in computer program requires special care some programming languages such as treat integer division as in case above so the answer is an integer other languages such as matlab and every computer algebra system return rational number as the answer as in case above these languages also provide functions to get the results of the other cases either directly or from the result of case names and symbols used for integer division include div and definitions vary regarding integer division when the dividend or the divisor is negative rounding may be toward zero so called division or toward division rarer styles can occur see modulo operation for the details divisibility rules can sometimes be used to quickly determine whether one integer divides exactly into another"}'
```


