#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).

    """

    import operator
    # Residual error = prediction - actual
    errors = [a-b for a,b in zip(predictions, net_worths)]
    # make data in form of tuple
    data = sorted(zip(ages, net_worths, errors),  key=lambda x: x[2], reverse=False)
    # sort data on error ascending
    #data = list(data)
    #data = data.sort(key=lambda x : x[2])
    # keep the first 90% of the sorted list ie get rid of the 10 points with highest residual errors
    cleaned_data = data[:int(len(predictions)*0.9)]

    return cleaned_data
