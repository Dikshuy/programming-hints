# python3

from collections import namedtuple

AssignedJob = namedtuple("AssignedJob", ["worker", "started_at"])

def SiftDown(i, size, data):
    min_idx = i
    l = 2*i + 1
    if l < size:
        if data[l][1] < data[min_idx][1]:
            min_idx = l
        elif data[l][1] == data[min_idx][1]:
            if data[l][0] < data[min_idx][0]:
                min_idx = l
    r = 2*i + 2
    if r < size:
        if data[r][1] < data[min_idx][1]:
            min_idx = r
        elif data[r][1] == data[min_idx][1]:
            if data[r][0] < data[min_idx][0]:
                min_idx = r

    if i != min_idx:
        data[i], data[min_idx] = data[min_idx], data[i]
        SiftDown(min_idx, size, data)

    return 

def assign_jobs(n_workers, jobs):
    finish_time = []
    assign_job = []
    for i in range(n_workers):
        finish_time.append([i, 0])

    for job in jobs:
        root = finish_time[0]
        start_worker = root[0]
        started_at = root[1]
        assign_job.append(AssignedJob(start_worker, started_at))
        finish_time[0][1] += job
        SiftDown(0, n_workers, finish_time)
    
    return assign_job


def main():
    n_workers, n_jobs = map(int, input().split())
    jobs = list(map(int, input().split()))
    assert len(jobs) == n_jobs

    assigned_jobs = assign_jobs(n_workers, jobs)

    for job in assigned_jobs:
        print(job.worker, job.started_at)


if __name__ == "__main__":
    main()
