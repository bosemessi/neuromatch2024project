import numpy as np
import pandas as pd


def annotate_licks(ls, rs, st):
    licks = ls.copy()
    rewards = rs.copy()
    stimulus_table = st.copy()

    bout_threshold = 0.7
    stim_start = stimulus_table[stimulus_table.omitted == False]["start_time"].values[0]
    stim_end = stimulus_table["start_time"].values[-1] + 0.75
    licks.query("(timestamps > @stim_start) and (timestamps <= @stim_end)", inplace=True)
    licks.reset_index(drop=True, inplace=True)
    # Computing ILI for each lick
    licks["pre_ili"] = np.concatenate([[np.nan], np.diff(licks.timestamps.values)])
    licks["post_ili"] = np.concatenate([np.diff(licks.timestamps.values), [np.nan]])
    # Segment licking bouts
    licks["bout_start"] = licks["pre_ili"] > bout_threshold
    licks["bout_end"] = licks["post_ili"] > bout_threshold
    licks.loc[licks["pre_ili"].apply(np.isnan), "bout_start"] = True
    licks.loc[licks["post_ili"].apply(np.isnan), "bout_end"] = True
    licks["bout_number"] = np.cumsum(licks["bout_start"])

    # Annotate rewards
    # Iterate through rewards
    licks["rewarded"] = False  # Setting default to False
    licks["num_rewards"] = 0
    for index, row in rewards.iterrows():
        if row["auto_rewarded"]:
            # Assign to nearest lick
            mylick = np.abs(licks.timestamps - row.timestamps).idxmin()
        else:
            # Assign reward to last lick before reward time
            this_reward_lick_times = np.where(licks.timestamps <= row.timestamps)[0]
            if len(this_reward_lick_times) == 0:
                raise Exception("First lick was after first reward")
            else:
                mylick = this_reward_lick_times[-1]
        licks.loc[mylick, "rewarded"] = True
        # licks can be double assigned to rewards because of auto-rewards
        licks.loc[mylick, "num_rewards"] += 1

    ## Annotate bout rewards
    x = (
        licks.groupby("bout_number")
        .any("rewarded")
        .reset_index()
        .rename(columns = {"rewarded": "bout_rewarded"})
    )
    y = (
        licks.groupby("bout_number")["num_rewards"]
        .sum()
        .reset_index()
        .rename(columns = {"num_rewards": "bout_num_rewards"})
    )

    temp = x[["bout_number", "bout_rewarded"]].merge(y, on="bout_number", how="left")
    licks = licks.merge(temp, on="bout_number", how="left")

    ## QC
    ## Check that all rewards are matched to a lick
    num_lick_rewards = licks["rewarded"].sum()
    num_rewards = len(rewards)
    double_rewards = np.sum(licks.query("num_rewards >1")["num_rewards"] - 1)
    assert (
        num_rewards == num_lick_rewards + double_rewards
    ), "Lick Annotations don't match number of rewards"

    # Check that all rewards are matched to a bout
    num_rewarded_bouts = np.sum(licks["bout_rewarded"] & licks["bout_start"])
    double_rewarded_bouts = np.sum(
        licks[
            licks["bout_rewarded"] & licks["bout_start"] & (licks["bout_num_rewards"] > 1)
        ]["bout_num_rewards"]
        - 1
    )
    assert (
        num_rewards == num_rewarded_bouts + double_rewarded_bouts
    ), "Bout Annotations don't match number of rewards"

    # Check that bouts start and stop
    num_bout_start = licks["bout_start"].sum()
    num_bout_end = licks["bout_end"].sum()
    num_bouts = licks["bout_number"].max()
    assert num_bout_start == num_bout_end, "Bout Starts and Bout Ends don't align"
    assert num_bout_start == num_bouts, "Number of bouts is incorrect"

    return licks


def annotate_bouts(ls, st):
    """
    Uses the bout annotations in licks to annotate stimulus_table

    Adds to stimulus_table
        bout_start,     (boolean) Whether a licking bout started during this image
        num_bout_start, (int) The number of licking bouts that started during this
                        image. This can be greater than 1 because the bout duration
                        is less than 750ms.
        bout_number,    (int) The label of the licking bout that started during this
                        image
        bout_end,       (boolean) Whether a licking bout ended during this image
        num_bout_end,   (int) The number of licking bouts that ended during this
                        image.

    """
    licks = ls.copy()
    stimulus_table = st.copy()

    # Annotate Bout Starts
    bout_starts = licks[licks["bout_start"]]
    stimulus_table["bout_start"] = False
    stimulus_table["num_bout_start"] = 0
    for index, x in bout_starts.iterrows():
        filter_start = stimulus_table.query("start_time < @x.timestamps")
        if len(filter_start) > 0:
            # Mark the last stimulus that started before the bout started
            start_index = filter_start.index[-1]
            stimulus_table.loc[start_index, "bout_start"] = True
            stimulus_table.loc[start_index, "num_bout_start"] += 1
            stimulus_table.loc[start_index, "bout_number"] = x.bout_number
        elif x.timestamps <= stimulus_table.iloc[0].start_time:
            # Bout started before stimulus, mark the first stimulus as start
            stimulus_table.loc[0, "bout_start"] = True
            stimulus_table.loc[0, "num_bout_start"] += 1
        else:
            raise Exception(
                "couldnt annotate bout start (bout number: {})".format(index)
            )

    # Annotate Bout Ends
    bout_ends = licks[licks["bout_end"]]
    stimulus_table["bout_end"] = False
    stimulus_table["num_bout_end"] = 0
    for index, x in bout_ends.iterrows():
        filter_end = stimulus_table.query("start_time < @x.timestamps")
        if len(filter_end) > 0:
            # Mark the last stimulus that started before the bout ended
            end_index = filter_end.index[-1]
            stimulus_table.loc[end_index, "bout_end"] = True
            stimulus_table.loc[end_index, "num_bout_end"] += 1
        elif x.timestamps <= stimulus_table.iloc[0].start_time:
            # Bout started before stimulus, mark the first stimulus as start
            stimulus_table.loc[0, "bout_end"] = True
            stimulus_table.loc[0, "num_bout_end"] += 1
        else:
            raise Exception("couldnt annotate bout end (bout number: {})".format(index))

    # Annotate In-Bout
    stimulus_table["in_lick_bout"] = (
        stimulus_table["num_bout_start"].cumsum()
        > stimulus_table["num_bout_end"].cumsum()
    )
    stimulus_table["in_lick_bout"] = stimulus_table["in_lick_bout"].shift(
        fill_value=False
    )
    overlap_index = (
        (stimulus_table["in_lick_bout"])
        & (stimulus_table["bout_start"])
        & (stimulus_table["num_bout_end"] >= 1)
    )
    stimulus_table.loc[overlap_index, "in_lick_bout"] = False

    stimulus_table = stimulus_table.merge(
        licks[["bout_number", "bout_start", "bout_end", "rewarded",	"num_rewards", "bout_rewarded",	"bout_num_rewards"]],
        on = ["bout_number", "bout_start", "bout_end"],
        how = "left"
    )
    

    ##### QC
    num_bouts_sp_start = stimulus_table["num_bout_start"].sum()
    num_bouts_sp_end = stimulus_table["num_bout_end"].sum()
    num_bouts_licks = licks.bout_start.sum()
    
    
    assert (
        num_bouts_sp_start == num_bouts_licks
    ), "Number of bouts doesnt match between licks table and stimulus table"
    assert (
        num_bouts_sp_start == num_bouts_sp_end
    ), "Mismatch between bout starts and bout ends"
    
    
    # assert stimulus_table.query("bout_start")[
    #     "licked"
    # ].all(), "All licking bout start should have licks"
    # assert stimulus_table.query("bout_end")[
    #     "licked"
    # ].all(), "All licking bout ends should have licks"
    # assert np.all(
    #     stimulus_table["in_lick_bout"]
    #     | stimulus_table["bout_start"]
    #     | ~stimulus_table["licked"]
    # ), "must either not have licked, or be in lick bout, or bout start"
    # assert np.all(
    #     ~(stimulus_table["in_lick_bout"] & stimulus_table["bout_start"])
    # ), "Cant be in a bout and a bout_start"

    return stimulus_table