def bucket_sort(input):
    #make buckets
    buckets =[[] for _ in range(len(input))]
    #assign values
    for value in input:
        bucket_index = int(value * len(input) // (max(input) + 1))
        buckets[bucket_index].append(value)

    #sort & merge
    sorted_list = []
    for bucket in buckets:
        result = quick_sort(bucket)
        sorted_list.extend(result)
    return sorted_list

def quick_sort(array):
    array_length = len(array)
    if(array_length <= 1):
        return array
    else:
        pivot = array[0]
        greater = [element for element in array[1:] if element > pivot]
        lesser = [element for element in array[1:] if element <= pivot]
        return quick_sort(lesser) + [pivot] + quick_sort(greater)

input = [9.9, 2.0, 5.5, 5.7, 3.4, 1.2, 8.3, 1.0, 4.4]
print(input)
print(bucket_sort(input))