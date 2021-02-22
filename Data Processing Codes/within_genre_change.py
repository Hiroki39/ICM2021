# within_genre_change.py
# illustrate the change within every genre over time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

music_df = pd.read_csv(
    '../2021_ICM_Problem_D_Data/full_music_data_with_scaled.csv')


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize=5):
            self.tick_params(pad=5)
            self.set_thetagrids(np.degrees(theta), labels=labels,
                                fontsize=fontsize)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.6, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def gen_case_data(genre):
    if genre == 'Overall':
        genre_music = music_df
    else:
        genre_music = music_df[music_df['genre'] == genre]
    parameter_columns = ['danceability', 'energy', 'valence',
                         'acousticness', 'instrumentalness', 'liveness',
                         'speechiness']
    case_data = np.empty((5, len(parameter_columns)))
    for i in range(5):
        target_music = genre_music[(genre_music['year'] > 1920 + i * 20) &
                                   (genre_music['year'] <= 1940 + i * 20)]
        case_data[i, :] = np.average(target_music[parameter_columns], axis=0)

    return case_data


parameters = ['danceability', 'energy', 'valence',
              'acousticness', 'liveness', 'instrumentalness',
              'speechiness']

genres = music_df['genre'].value_counts().index.tolist()
genres.remove('Unknown')
genres.append('Overall')

N = len(parameters)
theta = radar_factory(N)

fig, axes = plt.subplots(figsize=(20, 20), nrows=5, ncols=4,
                         subplot_kw=dict(projection='radar'))
fig.subplots_adjust(wspace=0.30, hspace=0.50, top=0.85, bottom=0.05)

colors = ['b', 'r', 'g', 'm', 'y']
# Plot the four cases from the example data on separate axes
for ax, genre in zip(axes.flat, genres):
    if not genre:
        continue
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], fontsize=10, angle=210)
    ax.set_title(genre, weight='bold', size=20, position=(0.5, 1.7),
                 horizontalalignment='center', verticalalignment='center')
    case_data = gen_case_data(genre)
    for d, color in zip(case_data, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25)
    ax.set_varlabels(parameters, fontsize=12)

# add legend relative to top-left plot
ax = axes[0, 3]
labels = ['1921~1940', '1941~1960', '1961~1980', '1981~2000', '2001~2020']
legend = ax.legend(labels, loc=(0.3, 1.35),
                   labelspacing=0.2, fontsize=15)

fig.suptitle('Change Of Each Genre Over Time', y=0.93,
             color='black', weight='bold',
             fontsize=30)

plt.show()
