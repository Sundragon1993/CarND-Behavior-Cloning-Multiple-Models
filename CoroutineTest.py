def fibonacci():
    numbers_list = []
    while 1:
        if (len(numbers_list) < 2):
            numbers_list.append(1)
        else:
            numbers_list.append(numbers_list[-1] + numbers_list[-2])
        yield numbers_list  # restart right after yield keyword
        # numbers_list.append(10)


our_generator = fibonacci()
my_output = []

for i in range(10):
    my_output = (next(our_generator))

print(my_output)


# def print_name(prefix):
#     print("Searching prefix:{}".format(prefix))
#     while True:
#         name = (yield)
#         if prefix in name:
#             print(name)
#
#         # calling coroutine, nothing will happen
#
#
# corou = print_name("Dear")
#
# # This will start execution of coroutine and
# # Prints first line "Searchig prefix..."
# # and advance execution to the first yield expression
# corou.__next__()
#
# # sending inputs
# corou.send("Atul")
# corou.send("Dear Atul")
# corou.send(" Atul")
# corou.send("Dear Atul1")
