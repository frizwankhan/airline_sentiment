# airline_sentiment

This repo has a basic sentiment analysis model that was trained on airline sentiment dataset. I have implemented a fast api to serve the model. The model is also containerised with docker and deployed in Heroku.

## Replicating the project
To replicate the project:
1. clone the repo
2. run `pip3 install -r requirements.txt`
3. run `python app.py` to run the api

you can make api request to http://0.0.0.0:5000/predict in your local machine, the request method must be post.
The Jason body should be in this format:
```
{
  "texts": ["sentence", "sentence]
}
```

To open swagger documentation open http://0.0.0.0:5000/docs

Here is the endpoint of heroku app:https://frizwankhan-airline-sentiment.herokuapp.com/
