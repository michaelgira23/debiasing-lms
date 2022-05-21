
import wandb

api = wandb.Api()

runs = api.runs("unprejudiced/unprejudiced")

for run_index, run in enumerate(runs):
    print(f'Process run {run_index + 1} of {len(runs)}')

    best_train_loss = float('inf')
    best_eval_loss = float('inf')

    best_icat = float('-inf')
    best_icat_index = float('-inf')
    best_train_loss_for_icat = float('inf')
    best_eval_loss_for_icat = float('inf')

    best_lm = float('-inf')
    best_lm_index = float('-inf')
    best_train_loss_for_lm = float('inf')
    best_eval_loss_for_lm = float('inf')

    best_ss = float('-inf')
    best_ss_index = float('-inf')
    best_train_loss_for_ss = float('inf')
    best_eval_loss_for_ss = float('inf')

    history = run.history(pandas=False)
    for index, sample in enumerate(history):
        try:
            sample_train_loss = sample['train_loss']
            sample_eval_loss = sample['eval_loss']

            sample_icat = sample['intrasentence_overall_ICAT Score']
            sample_lm = sample['intrasentence_overall_LM Score']
            sample_ss = sample['intrasentence_overall_SS Score']
        except:
            print(
                f'Skip run {run_index + 1} because doesn\'t contain sample key')
            continue

        best_train_loss = min(best_train_loss, sample_train_loss)
        best_eval_loss = min(best_eval_loss, sample_eval_loss)

        if sample_icat > best_icat:
            best_icat = sample_icat
            best_icat_index = index
            best_train_loss_for_icat = sample_train_loss
            best_eval_loss_for_icat = sample_eval_loss

        if sample_lm > best_lm:
            best_lm = sample_lm
            best_lm_index = index
            best_train_loss_for_lm = sample_train_loss
            best_eval_loss_for_lm = sample_eval_loss

        if abs(sample_ss - 50) < abs(best_ss - 50):
            best_ss = sample_ss
            best_ss_index = index
            best_train_loss_for_ss = sample_train_loss
            best_eval_loss_for_ss = sample_eval_loss

    run.summary['min_train_loss'] = best_train_loss
    run.summary['min_eval_loss'] = best_eval_loss

    run.summary['best_icat_index'] = best_icat_index
    run.summary['best_train_loss_for_icat'] = best_train_loss_for_icat
    run.summary['best_eval_loss_for_icat'] = best_eval_loss_for_icat

    run.summary['best_lm_index'] = best_lm_index
    run.summary['best_train_loss_for_lm'] = best_train_loss_for_lm
    run.summary['best_eval_loss_for_lm'] = best_eval_loss_for_lm

    run.summary['best_ss_index'] = best_ss_index
    run.summary['best_train_loss_for_ss'] = best_train_loss_for_ss
    run.summary['best_eval_loss_for_ss'] = best_eval_loss_for_ss

    run.update()
