import torch

from scripts.pod_eval_vllm import compute_kl_from_sparse, dense_to_sparse_topk


def test_tail_bucket_penalizes_probability_mass_outside_teacher_topk():
    teacher_logits = torch.tensor([[[4.0, 3.0, 0.0, 0.0]]])
    sparse = dense_to_sparse_topk(teacher_logits, k=2)

    student_same_topk_bad_tail = torch.tensor([[[4.0, 3.0, 6.0, 6.0]]])

    legacy = compute_kl_from_sparse(
        sparse["indices"],
        sparse["values"],
        student_same_topk_bad_tail,
        teacher_logsumexp=sparse["logsumexp"],
        mode="conditional",
    )
    tail_bucket = compute_kl_from_sparse(
        sparse["indices"],
        sparse["values"],
        student_same_topk_bad_tail,
        teacher_logsumexp=sparse["logsumexp"],
        mode="tail_bucket",
    )

    assert legacy.item() < 1e-6
    assert tail_bucket.item() > 2.0


def test_tail_bucket_matches_full_kl_when_topk_covers_vocab():
    teacher_logits = torch.tensor([[[2.0, 0.0, -1.0, -3.0]]])
    student_logits = torch.tensor([[[1.5, 0.5, -0.5, -2.5]]])
    sparse = dense_to_sparse_topk(teacher_logits, k=4)

    tail_bucket = compute_kl_from_sparse(
        sparse["indices"],
        sparse["values"],
        student_logits,
        teacher_logsumexp=sparse["logsumexp"],
        mode="tail_bucket",
    )
    t_log_p = torch.log_softmax(teacher_logits, dim=-1)
    s_log_p = torch.log_softmax(student_logits, dim=-1)
    full = (t_log_p.exp() * (t_log_p - s_log_p)).sum(dim=-1)

    assert torch.allclose(tail_bucket, full, atol=1e-6)
