import torch

from gpt2_beam import _add_beam_candidate

def test_add_beam_candidate():
    best_score = {}
    best_sequence = torch.zeros((1, 64), dtype=torch.int64)
    batch_size = 1
    num_beams = 10
    beam_scores_array = [-1.0673, -1.8507, -1.8664, -2.3499, -2.8584, -3.5715, -3.6126, -4.3180, -4.3217, -4.8584]
    beam_scores = torch.tensor(beam_scores_array).reshape(1, 10)
    history_array = [4518, 1318, 554, 383, 40764, 317, 1002, 612, 1052, 7754]
    history = torch.tensor(history_array).reshape(10, 1)
    eos_token_id = [50256, 628]

    _add_beam_candidate(
        best_score,
        best_sequence,
        batch_size,
        num_beams,
        beam_scores,
        history,
        eos_token_id
    )

    assert best_score == {}
    assert torch.equal(best_sequence, torch.zeros((1, 64), dtype=torch.int64))


