import matplotlib.pyplot as plt
import numpy as np

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')


def live_plotter(x_vec, y_data, lines, labels, identifier='', pause_time=0.1):
    if not lines[0]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        for i in range(len(lines)):
            # create a variable for the line so we can later update it
            lines[i], = ax.plot(x_vec, y_data[i], '-o', alpha=0.8, label=labels[i])
        ax.legend()
        # update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    total_max = np.max(y_data[0])
    total_max_num = 0
    total_min = np.min(y_data[0])
    total_min_num = 0
    for i in range(len(lines)):

        # after the figure, axis, and line are created, we only need to update the y-data
        y_data_value = y_data[i]
        lines[i].set_ydata(y_data[i])
        # adjust limits if new data goes beyond bounds
        if np.min(y_data_value) <= lines[i].axes.get_ylim()[0]:
            plt.ylim([np.min(y_data_value) - np.std(y_data_value), lines[i].axes.get_ylim()[1]])

        if np.max(y_data_value) >= lines[i].axes.get_ylim()[1]:
            plt.ylim([lines[i].axes.get_ylim()[0], np.max(y_data_value) + np.std(y_data_value)])

        if total_max < np.max(y_data_value):
            total_max = np.max(y_data_value)
            total_max_num = i

        if total_min > np.min(y_data_value):
            total_min = np.min(y_data_value)
            total_min_num = i

    shift_amount = ((lines[0].axes.get_ylim()[1]) - (lines[0].axes.get_ylim()[0])) * 0.75
    if total_max <= lines[0].axes.get_ylim()[1] - shift_amount:
        plt.ylim([lines[0].axes.get_ylim()[0], np.max(y_data[total_max_num]) + np.std(y_data[total_max_num])])

    if total_min >= lines[0].axes.get_ylim()[0] + shift_amount:
        plt.ylim([np.min(y_data[total_min_num]) - np.std(y_data[total_min_num]), lines[0].axes.get_ylim()[1]])

    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return lines

#
# size = 100
# x_vec = np.linspace(0, 1, size + 1)[0:-1]
# y_vec = [np.random.randn(len(x_vec)), np.random.randn(len(x_vec)), np.random.randn(len(x_vec))]
# lines = [[], [], []]
# counter = 0
# while True:
#     y_vec[0][-1] = abs(np.random.randn(1))*counter / 10
#     y_vec[1][-1] = np.random.randn(1)
#     y_vec[2][-1] = np.random.randn(1)
#     lines = live_plotter(x_vec, y_vec, lines)
#     y_vec[0] = np.append(y_vec[0][1:], 0.0)
#     y_vec[1] = np.append(y_vec[1][1:], 0.0)
#     y_vec[2] = np.append(y_vec[2][1:], 0.0)
#     counter += 1
