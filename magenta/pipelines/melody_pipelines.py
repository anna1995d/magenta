# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data processing pipelines for melodies."""

from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from note_seq import events_lib
from note_seq import melodies_lib
from note_seq import Melody
from note_seq import PolyphonicMelodyError
from note_seq import sequences_lib
from note_seq.protobuf import music_pb2
import numpy as np
import tensorflow.compat.v1 as tf

# monkey Melody.from_quantized_sequence()
# def new_from_quantized_sequence(self,
#                               quantized_sequence,
#                               mode = 'standard',
#                               search_start_step=0,
#                               instrument=0,
#                               gap_bars=1,
#                               ignore_polyphonic_notes=False,
#                               pad_end=False,
#                               filter_drums=True):
#     """Populate self with a melody from the given quantized NoteSequence.
#     A monophonic melody is extracted from the given `instrument` starting at
#     `search_start_step`. `instrument` and `search_start_step` can be used to
#     drive extraction of multiple melodies from the same quantized sequence. The
#     end step of the extracted melody will be stored in `self._end_step`.
#     0 velocity notes are ignored. The melody extraction is ended when there are
#     no held notes for a time stretch of `gap_bars` in bars (measures) of music.
#     The number of time steps per bar is computed from the time signature in
#     `quantized_sequence`.
#     `ignore_polyphonic_notes` determines what happens when polyphonic (multiple
#     notes start at the same time) data is encountered. If
#     `ignore_polyphonic_notes` is true, the highest pitch is used in the melody
#     when multiple notes start at the same time. If false, an exception is
#     raised.
#     Args:
#       quantized_sequence: A NoteSequence quantized with
#           sequences_lib.quantize_note_sequence.
#       search_start_step: Start searching for a melody at this time step. Assumed
#           to be the first step of a bar.
#       instrument: Search for a melody in this instrument number.
#       gap_bars: If this many bars or more follow a NOTE_OFF event, the melody
#           is ended.
#       ignore_polyphonic_notes: If True, the highest pitch is used in the melody
#           when multiple notes start at the same time. If False,
#           PolyphonicMelodyError will be raised if multiple notes start at
#           the same time.
#       pad_end: If True, the end of the melody will be padded with NO_EVENTs so
#           that it will end at a bar boundary.
#       filter_drums: If True, notes for which `is_drum` is True will be ignored.
#     Raises:
#       NonIntegerStepsPerBarError: If `quantized_sequence`'s bar length
#           (derived from its time signature) is not an integer number of time
#           steps.
#       PolyphonicMelodyError: If any of the notes start on the same step
#           and `ignore_polyphonic_notes` is False.
#     """
#     sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)
#     self._reset()

#     steps_per_bar_float = sequences_lib.steps_per_bar_in_quantized_sequence(
#         quantized_sequence)
#     if steps_per_bar_float % 1 != 0:
#       raise events_lib.NonIntegerStepsPerBarError(
#           'There are %f timesteps per bar. Time signature: %d/%d' %
#           (steps_per_bar_float, quantized_sequence.time_signatures[0].numerator,
#            quantized_sequence.time_signatures[0].denominator))
#     self._steps_per_bar = steps_per_bar = int(steps_per_bar_float)
#     self._steps_per_quarter = (
#         quantized_sequence.quantization_info.steps_per_quarter)

#     # Sort track by note start times, and secondarily by pitch descending.
#     notes = sorted([n for n in quantized_sequence.notes
#                     if n.instrument == instrument and
#                     n.quantized_start_step >= search_start_step],
#                    key=lambda note: (note.quantized_start_step, -note.pitch))

#     if not notes:
#       return

#     # The first step in the melody, beginning at the first step of a bar.
#     melody_start_step = (
#         notes[0].quantized_start_step -
#         (notes[0].quantized_start_step - search_start_step) % steps_per_bar)
#     for note in notes:
#       if filter_drums and note.is_drum:
#         continue

#       # Ignore 0 velocity notes.
#       if not note.velocity:
#         continue

#       start_index = note.quantized_start_step - melody_start_step
#       end_index = note.quantized_end_step - melody_start_step

#       if not self._events:
#         # If there are no events, we don't need to check for polyphony.
#         self._add_note(note.pitch, start_index, end_index)
#         continue

#       # If `start_index` comes before or lands on an already added note's start
#       # step, we cannot add it. In that case either discard the melody or keep
#       # the highest pitch.
#       last_on, last_off = self._get_last_on_off_events()
#       on_distance = start_index - last_on
#       off_distance = start_index - last_off
#       if on_distance == 0:
#         if ignore_polyphonic_notes:
#           # Keep highest note.
#           # Notes are sorted by pitch descending, so if a note is already at
#           # this position its the highest pitch.
#           continue
#         else:
#           self._reset()
#           raise PolyphonicMelodyError()
#       elif on
#       elif on_distance < 0:
#         raise PolyphonicMelodyError(
#             'Unexpected note. Not in ascending order.')

#       # If a gap of `gap` or more steps is found, end the melody.
#       if len(self) and off_distance >= gap_bars * steps_per_bar:  # pylint:disable=len-as-condition
#         break

#       # Add the note-on and off events to the melody.
#       self._add_note(note.pitch, start_index, end_index)

#     if not self._events:
#       # If no notes were added, don't set `_start_step` and `_end_step`.
#       return

#     self._start_step = melody_start_step

#     # Strip final MELODY_NOTE_OFF event.
#     if self._events[-1] == MELODY_NOTE_OFF:
#       del self._events[-1]

#     length = len(self)
#     # Optionally round up `_end_step` to a multiple of `steps_per_bar`.
#     if pad_end:
#       length += -len(self) % steps_per_bar
#     self.set_length(length)

class MelodyExtractor(pipeline.Pipeline):
  """Extracts monophonic melodies from a quantized NoteSequence."""

  def __init__(self, min_bars=7, max_steps=512, min_unique_pitches=5,
               gap_bars=1.0, ignore_polyphonic_notes=False, filter_drums=True,
               name=None):
    super(MelodyExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=melodies_lib.Melody,
        name=name)
    self._min_bars = min_bars
    self._max_steps = max_steps
    self._min_unique_pitches = min_unique_pitches
    self._gap_bars = gap_bars
    self._ignore_polyphonic_notes = ignore_polyphonic_notes
    self._filter_drums = filter_drums

  def transform(self, input_object):
    quantized_sequence = input_object
    try:
      melodies, stats = extract_melodies(
          quantized_sequence,
          min_bars=self._min_bars,
          max_steps_truncate=self._max_steps,
          min_unique_pitches=self._min_unique_pitches,
          gap_bars=self._gap_bars,
          ignore_polyphonic_notes=self._ignore_polyphonic_notes,
          filter_drums=self._filter_drums)
    except events_lib.NonIntegerStepsPerBarError as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      melodies = []
      stats = [statistics.Counter('non_integer_steps_per_bar', 1)]
    self._set_stats(stats)
    return melodies

def extract_melodies(quantized_sequence,
                     mel_mode='standard',
                     search_start_step=0,
                     min_bars=7,
                     max_steps_truncate=None,
                     max_steps_discard=None,
                     gap_bars=1.0,
                     min_unique_pitches=5,
                     ignore_polyphonic_notes=True,
                     pad_end=False,
                     filter_drums=True):
  """Extracts a list of melodies from the given quantized NoteSequence.

  This function will search through `quantized_sequence` for monophonic
  melodies in every track at every time step.

  Once a note-on event in a track is encountered, a melody begins.
  Gaps of silence in each track will be splitting points that divide the
  track into separate melodies. The minimum size of these gaps are given
  in `gap_bars`. The size of a bar (measure) of music in time steps is
  computed from the time signature stored in `quantized_sequence`.

  The melody is then checked for validity. The melody is only used if it is
  at least `min_bars` bars long, and has at least `min_unique_pitches` unique
  notes (preventing melodies that only repeat a few notes, such as those found
  in some accompaniment tracks, from being used).

  After scanning each instrument track in the quantized sequence, a list of all
  extracted Melody objects is returned.

  Args:
    quantized_sequence: A quantized NoteSequence.
    mode: 'standard', 'skyline1', 'skyline2'
    search_start_step: Start searching for a melody at this time step. Assumed
        to be the first step of a bar.
    min_bars: Minimum length of melodies in number of bars. Shorter melodies are
        discarded.
    max_steps_truncate: Maximum number of steps in extracted melodies. If
        defined, longer melodies are truncated to this threshold. If pad_end is
        also True, melodies will be truncated to the end of the last bar below
        this threshold.
    max_steps_discard: Maximum number of steps in extracted melodies. If
        defined, longer melodies are discarded.
    gap_bars: A melody comes to an end when this number of bars (measures) of
        silence is encountered.
    min_unique_pitches: Minimum number of unique notes with octave equivalence.
        Melodies with too few unique notes are discarded.
    ignore_polyphonic_notes: If True, melodies will be extracted from
        `quantized_sequence` tracks that contain polyphony (notes start at
        the same time). If False, tracks with polyphony will be ignored.
    pad_end: If True, the end of the melody will be padded with NO_EVENTs so
        that it will end at a bar boundary.
    filter_drums: If True, notes for which `is_drum` is True will be ignored.

  Returns:
    melodies: A python list of Melody instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.

  Raises:
    NonIntegerStepsPerBarError: If `quantized_sequence`'s bar length
        (derived from its time signature) is not an integer number of time
        steps.
  """
  sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)

  # TODO(danabo): Convert `ignore_polyphonic_notes` into a float which controls
  # the degree of polyphony that is acceptable.
  melodies = []
  # pylint: disable=g-complex-comprehension
  stats = dict((stat_name, statistics.Counter(stat_name)) for stat_name in
               ['polyphonic_tracks_discarded',
                'melodies_discarded_too_short',
                'melodies_discarded_too_few_pitches',
                'melodies_discarded_too_long',
                'melodies_truncated'])
  # pylint: enable=g-complex-comprehension
  # Create a histogram measuring melody lengths (in bars not steps).
  # Capture melodies that are very small, in the range of the filter lower
  # bound `min_bars`, and large. The bucket intervals grow approximately
  # exponentially.
  stats['melody_lengths_in_bars'] = statistics.Histogram(
      'melody_lengths_in_bars',
      [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, min_bars // 2, min_bars,
       min_bars + 1, min_bars - 1])
  instruments = set(n.instrument for n in quantized_sequence.notes)
  steps_per_bar = int(
      sequences_lib.steps_per_bar_in_quantized_sequence(quantized_sequence))

  if mel_mode=='standard':
    # the original extraction method 
    for instrument in instruments:
      instrument_search_start_step = search_start_step
      # Quantize the track into a Melody object.
      # If any notes start at the same time, only one is kept.
      while 1:
        melody = Melody()
        try:
          melody.from_quantized_sequence(
              quantized_sequence,
              # mode=mode,
              instrument=instrument,
              search_start_step=instrument_search_start_step,
              gap_bars=gap_bars,
              ignore_polyphonic_notes=ignore_polyphonic_notes,
              pad_end=pad_end,
              filter_drums=filter_drums)
        except PolyphonicMelodyError:
          stats['polyphonic_tracks_discarded'].increment()
          break  # Look for monophonic melodies in other tracks.
        # Start search for next melody on next bar boundary (inclusive).
        instrument_search_start_step = (
            melody.end_step +
            (search_start_step - melody.end_step) % steps_per_bar)
        if not melody:
          break

        # Require a certain melody length.
        if len(melody) < melody.steps_per_bar * min_bars:
          stats['melodies_discarded_too_short'].increment()
          continue

        # Discard melodies that are too long.
        if max_steps_discard is not None and len(melody) > max_steps_discard:
          stats['melodies_discarded_too_long'].increment()
          continue

        # Truncate melodies that are too long.
        if max_steps_truncate is not None and len(melody) > max_steps_truncate:
          truncated_length = max_steps_truncate
          if pad_end:
            truncated_length -= max_steps_truncate % melody.steps_per_bar
          melody.set_length(truncated_length)
          stats['melodies_truncated'].increment()

        # Require a certain number of unique pitches.
        note_histogram = melody.get_note_histogram()
        unique_pitches = np.count_nonzero(note_histogram)
        # print("unique pitches", unique_pitches)
        if unique_pitches < min_unique_pitches:
          stats['melodies_discarded_too_few_pitches'].increment()
          continue

        # TODO(danabo)
        # Add filter for rhythmic diversity.

        stats['melody_lengths_in_bars'].increment(
            len(melody) // melody.steps_per_bar)

        melodies.append(melody)

    return melodies, list(stats.values())
  
  elif mel_mode=='skyline1':
    # Skyline extraction that discards polyphonic notes
    instrument = 0
    for n in quantized_sequence.notes:
      n.instrument = instrument
    current_search_start_step = search_start_step

    while 1:
      melody = Melody()
      try:
        melody.from_quantized_sequence(
            quantized_sequence,
            # mode=mode,
            instrument=instrument,
            search_start_step=current_search_start_step,
            gap_bars=gap_bars,
            ignore_polyphonic_notes=ignore_polyphonic_notes,
            pad_end=pad_end,
            filter_drums=filter_drums)
      except PolyphonicMelodyError:
        stats['polyphonic_tracks_discarded'].increment()
        break  # Look for monophonic melodies in other tracks.
      # Start search for next melody on next bar boundary (inclusive).
      current_search_start_step = (
          melody.end_step +
          (search_start_step - melody.end_step) % steps_per_bar)
      if not melody:
        break

      # Require a certain melody length.
      if len(melody) < melody.steps_per_bar * min_bars:
        stats['melodies_discarded_too_short'].increment()
        continue

      # Discard melodies that are too long.
      if max_steps_discard is not None and len(melody) > max_steps_discard:
        stats['melodies_discarded_too_long'].increment()
        continue

      # Truncate melodies that are too long.
      if max_steps_truncate is not None and len(melody) > max_steps_truncate:
        truncated_length = max_steps_truncate
        if pad_end:
          truncated_length -= max_steps_truncate % melody.steps_per_bar
        melody.set_length(truncated_length)
        stats['melodies_truncated'].increment()

      # Require a certain number of unique pitches.
      note_histogram = melody.get_note_histogram()
      unique_pitches = np.count_nonzero(note_histogram)
      # print("unique pitches", unique_pitches)
      if unique_pitches < min_unique_pitches:
        stats['melodies_discarded_too_few_pitches'].increment()
        continue

      # TODO(danabo)
      # Add filter for rhythmic diversity.

      stats['melody_lengths_in_bars'].increment(
          len(melody) // melody.steps_per_bar)

      melodies.append(melody)
  # elif mode=='skyline2':
  #   # Skyline extraction that partially includes polyphonic notes
  #   instrument = 0
  #   for n in quantized_sequence.notes:
  #     n.instrument = instrument
  #   current_search_start_step = search_start_step

  #   while 1:
  #     melody = Melody()
  #     try:
  #       melody.from_quantized_sequence(
  #           quantized_sequence,
  #           # mode=mode,
  #           instrument=instrument,
  #           search_start_step=current_search_start_step,
  #           gap_bars=gap_bars,
  #           ignore_polyphonic_notes=ignore_polyphonic_notes,
  #           pad_end=pad_end,
  #           filter_drums=filter_drums)
  #     except PolyphonicMelodyError:
  #       stats['polyphonic_tracks_discarded'].increment()
  #       break  # Look for monophonic melodies in other tracks.
  #     # Start search for next melody on next bar boundary (inclusive).
  #     current_search_start_step = (
  #         melody.end_step +
  #         (search_start_step - melody.end_step) % steps_per_bar)
  #     if not melody:
  #       break

  #     # Require a certain melody length.
  #     if len(melody) < melody.steps_per_bar * min_bars:
  #       stats['melodies_discarded_too_short'].increment()
  #       continue

  #     # Discard melodies that are too long.
  #     if max_steps_discard is not None and len(melody) > max_steps_discard:
  #       stats['melodies_discarded_too_long'].increment()
  #       continue

  #     # Truncate melodies that are too long.
  #     if max_steps_truncate is not None and len(melody) > max_steps_truncate:
  #       truncated_length = max_steps_truncate
  #       if pad_end:
  #         truncated_length -= max_steps_truncate % melody.steps_per_bar
  #       melody.set_length(truncated_length)
  #       stats['melodies_truncated'].increment()

  #     # Require a certain number of unique pitches.
  #     note_histogram = melody.get_note_histogram()
  #     unique_pitches = np.count_nonzero(note_histogram)
  #     if unique_pitches < min_unique_pitches:
  #       stats['melodies_discarded_too_few_pitches'].increment()
  #       continue

  #     # TODO(danabo)
  #     # Add filter for rhythmic diversity.

  #     stats['melody_lengths_in_bars'].increment(
  #         len(melody) // melody.steps_per_bar)

  #     melodies.append(melody)

  #   return melodies, list(stats.values())

  else:
    raise 'Invalid mode of extraction.'

  # print("-------------\n")
  # for x in stats.keys():
  #   stats[x]._pretty_print(x)
  # print("-------------\n")
  return melodies, list(stats.values())




# def extract_melodies(quantized_sequence,
#                      search_start_step=0,
#                      min_bars=7,
#                      max_steps_truncate=None,
#                      max_steps_discard=None,
#                      gap_bars=1.0,
#                      min_unique_pitches=5,
#                      ignore_polyphonic_notes=True,
#                      pad_end=False,
#                      filter_drums=True):
#   """Extracts a list of melodies from the given quantized NoteSequence.

#   This function will search through `quantized_sequence` for monophonic
#   melodies in every track at every time step.

#   Once a note-on event in a track is encountered, a melody begins.
#   Gaps of silence in each track will be splitting points that divide the
#   track into separate melodies. The minimum size of these gaps are given
#   in `gap_bars`. The size of a bar (measure) of music in time steps is
#   computed from the time signature stored in `quantized_sequence`.

#   The melody is then checked for validity. The melody is only used if it is
#   at least `min_bars` bars long, and has at least `min_unique_pitches` unique
#   notes (preventing melodies that only repeat a few notes, such as those found
#   in some accompaniment tracks, from being used).

#   After scanning each instrument track in the quantized sequence, a list of all
#   extracted Melody objects is returned.

#   Args:
#     quantized_sequence: A quantized NoteSequence.
#     search_start_step: Start searching for a melody at this time step. Assumed
#         to be the first step of a bar.
#     min_bars: Minimum length of melodies in number of bars. Shorter melodies are
#         discarded.
#     max_steps_truncate: Maximum number of steps in extracted melodies. If
#         defined, longer melodies are truncated to this threshold. If pad_end is
#         also True, melodies will be truncated to the end of the last bar below
#         this threshold.
#     max_steps_discard: Maximum number of steps in extracted melodies. If
#         defined, longer melodies are discarded.
#     gap_bars: A melody comes to an end when this number of bars (measures) of
#         silence is encountered.
#     min_unique_pitches: Minimum number of unique notes with octave equivalence.
#         Melodies with too few unique notes are discarded.
#     ignore_polyphonic_notes: If True, melodies will be extracted from
#         `quantized_sequence` tracks that contain polyphony (notes start at
#         the same time). If False, tracks with polyphony will be ignored.
#     pad_end: If True, the end of the melody will be padded with NO_EVENTs so
#         that it will end at a bar boundary.
#     filter_drums: If True, notes for which `is_drum` is True will be ignored.

#   Returns:
#     melodies: A python list of Melody instances.
#     stats: A dictionary mapping string names to `statistics.Statistic` objects.

#   Raises:
#     NonIntegerStepsPerBarError: If `quantized_sequence`'s bar length
#         (derived from its time signature) is not an integer number of time
#         steps.
#   """
#   sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)

#   # TODO(danabo): Convert `ignore_polyphonic_notes` into a float which controls
#   # the degree of polyphony that is acceptable.
#   melodies = []
#   # pylint: disable=g-complex-comprehension
#   stats = dict((stat_name, statistics.Counter(stat_name)) for stat_name in
#                ['polyphonic_tracks_discarded',
#                 'melodies_discarded_too_short',
#                 'melodies_discarded_too_few_pitches',
#                 'melodies_discarded_too_long',
#                 'melodies_truncated'])
#   # pylint: enable=g-complex-comprehension
#   # Create a histogram measuring melody lengths (in bars not steps).
#   # Capture melodies that are very small, in the range of the filter lower
#   # bound `min_bars`, and large. The bucket intervals grow approximately
#   # exponentially.
#   stats['melody_lengths_in_bars'] = statistics.Histogram(
#       'melody_lengths_in_bars',
#       [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, min_bars // 2, min_bars,
#        min_bars + 1, min_bars - 1])
#   instruments = set(n.instrument for n in quantized_sequence.notes)
#   steps_per_bar = int(
#       sequences_lib.steps_per_bar_in_quantized_sequence(quantized_sequence))
#   for instrument in instruments:
#     instrument_search_start_step = search_start_step
#     # Quantize the track into a Melody object.
#     # If any notes start at the same time, only one is kept.
#     while 1:
#       melody = Melody()
#       try:
#         melody.from_quantized_sequence(
#             quantized_sequence,
#             instrument=instrument,
#             search_start_step=instrument_search_start_step,
#             gap_bars=gap_bars,
#             ignore_polyphonic_notes=ignore_polyphonic_notes,
#             pad_end=pad_end,
#             filter_drums=filter_drums)
#       except PolyphonicMelodyError:
#         stats['polyphonic_tracks_discarded'].increment()
#         break  # Look for monophonic melodies in other tracks.
#       # Start search for next melody on next bar boundary (inclusive).
#       instrument_search_start_step = (
#           melody.end_step +
#           (search_start_step - melody.end_step) % steps_per_bar)
#       if not melody:
#         break

#       # Require a certain melody length.
#       if len(melody) < melody.steps_per_bar * min_bars:
#         stats['melodies_discarded_too_short'].increment()
#         continue

#       # Discard melodies that are too long.
#       if max_steps_discard is not None and len(melody) > max_steps_discard:
#         stats['melodies_discarded_too_long'].increment()
#         continue

#       # Truncate melodies that are too long.
#       if max_steps_truncate is not None and len(melody) > max_steps_truncate:
#         truncated_length = max_steps_truncate
#         if pad_end:
#           truncated_length -= max_steps_truncate % melody.steps_per_bar
#         melody.set_length(truncated_length)
#         stats['melodies_truncated'].increment()

#       # Require a certain number of unique pitches.
#       note_histogram = melody.get_note_histogram()
#       unique_pitches = np.count_nonzero(note_histogram)
#       if unique_pitches < min_unique_pitches:
#         stats['melodies_discarded_too_few_pitches'].increment()
#         continue

#       # TODO(danabo)
#       # Add filter for rhythmic diversity.

#       stats['melody_lengths_in_bars'].increment(
#           len(melody) // melody.steps_per_bar)

#       melodies.append(melody)

#   return melodies, list(stats.values())


