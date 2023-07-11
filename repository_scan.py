import csv
import shutil
import sys

import git
import pandas as pd
from pydriller import Repository, Git
from pydriller.metrics.process.change_set import ChangeSet
from pydriller.metrics.process.code_churn import CodeChurn
from pydriller.metrics.process.commits_count import CommitsCount
from pydriller.metrics.process.contributors_count import ContributorsCount
from pydriller.metrics.process.contributors_experience import ContributorsExperience
from pydriller.metrics.process.hunks_count import HunksCount
from pydriller.metrics.process.lines_count import LinesCount
from github import Github


def metric_calculation_and_writing(start, to, commit_link, writer, label):
    print("Metrics calculation...")
    code_churn = CodeChurn(path_to_repo=commit_link,
                           from_commit=start,
                           to_commit=to)
    files_count = code_churn.count()
    files_max = code_churn.max()
    files_avg = code_churn.avg()
    filename = files_count.keys()

    commits_count = CommitsCount(path_to_repo=commit_link,
                                 from_commit=start,
                                 to_commit=to)
    num_commit = commits_count.count()

    contributors_count = ContributorsCount(path_to_repo=commit_link,
                                           from_commit=start,
                                           to_commit=to)
    count = contributors_count.count()
    minor = contributors_count.count_minor()

    contributors_experience = ContributorsExperience(path_to_repo=commit_link,
                                                     from_commit=start,
                                                     to_commit=to)
    contr_exp = contributors_experience.count()

    hunks_count = HunksCount(path_to_repo=commit_link,
                             from_commit=start,
                             to_commit=to)
    num_hunks = hunks_count.count()

    lines_count = LinesCount(path_to_repo=commit_link,
                             from_commit=start,
                             to_commit=to)

    added_count = lines_count.count_added()
    added_max = lines_count.max_added()
    added_avg = lines_count.avg_added()

    removed_count = lines_count.count_removed()
    removed_max = lines_count.max_removed()
    removed_avg = lines_count.avg_removed()

    for file in filename:
        if file is None:
            continue
        if label == -1:
            writer.writerow([file, files_count.get(file), files_max.get(file), files_avg.get(file), num_commit.get(file),
                             count.get(file), minor.get(file), contr_exp.get(file), num_hunks.get(file),
                             added_count.get(file), added_max.get(file), added_avg.get(file), removed_count.get(file),
                             removed_max.get(file), removed_avg.get(file)])
        else:
            writer.writerow(
                [file, files_count.get(file), files_max.get(file), files_avg.get(file), num_commit.get(file),
                 count.get(file), minor.get(file), contr_exp.get(file), num_hunks.get(file),
                 added_count.get(file), added_max.get(file), added_avg.get(file), removed_count.get(file),
                 removed_max.get(file), removed_avg.get(file), label])


def get_commit_count(repo_link):

    parts = repo_link.split("/")
    username = parts[-2]
    repository = parts[-1]

    g = Github()
    repo = g.get_repo(f"{username}/{repository}")
    commit_count = repo.get_commits().totalCount

    return commit_count


def main():
    cve = pd.read_csv("./data/CVEfixes.csv")
    filename = 'dataset.csv'
    # skipped_repos = 0

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', "#CodeChurnInFile",
                         "MaxSizeCodeChurn", "AvgSizeCodeChurn", "#Commits", "#Contributors",
                         "#MinorContributors", "#ContibutorExperience", "#Hunks", "#LinesAdded", "MaxLinesAdded",
                         "AvgLinesAdded", "#LinesRemoved", "MaxLinesRemoved", "AvgLinesRemoved", "vulnerable"])

        for i in range(len(cve)):
            try:
                # Retrieving commit link and its fixed and vulnerable commits hash
                commit_link = cve.iloc[i]['repository']
                fixed_hash = cve.iloc[i]['fixed_hash']
                vulnerable_hash = None
                pre_fix_hash = None

                print("\n\nRepo link: ", commit_link)

                # for commit in Repository(repo_link, single=vulnerable_hash).traverse_commits():
                #     print(commit.dmm_unit_size)
                #     print(commit.dmm_unit_complexity)
                #     print(commit.dmm_unit_interfacing)
                #
                # for commit in Repository(repo_link, single=fixed_hash).traverse_commits():
                #     print(commit.dmm_unit_size)
                #     print(commit.dmm_unit_complexity)
                #     print(commit.dmm_unit_interfacing)

                # Excluding repos with too many commits
                # num_commit = get_commit_count(commit_link)
                # if num_commit > 10000:
                #     print("\nSkipped repo with", num_commit, "commits")
                #     skipped_repos = skipped_repos + 1
                #     continue

                repository = Repository(commit_link, single=fixed_hash)
                for commit in repository.traverse_commits():
                    # commit vulnerabile
                    print("Finding the most obsolete vulnerable commit")
                    buggy_commits = repository.git.get_commits_last_modified_lines(commit)
                    if len(buggy_commits) > 0:
                        print(buggy_commits)
                    else:
                        print("Unable to find most obsolete vulnerable commit")

                    # Scanning the obtained commits to find the most obsolete one
                    older_vulnerable_commit = None
                    for key, value in buggy_commits.items():
                        for index in range(len(value)):
                            vulnerable_commit = repository.git.get_commit(value.pop())
                            if older_vulnerable_commit is None:
                                older_vulnerable_commit = vulnerable_commit
                                continue
                            print("Data commit: ", vulnerable_commit.committer_date,
                                  "Data older: ", older_vulnerable_commit.committer_date)
                            data_older = older_vulnerable_commit.committer_date
                            data_new = vulnerable_commit.committer_date
                            if data_new < data_older:
                                older_vulnerable_commit = vulnerable_commit

                    vulnerable_hash = older_vulnerable_commit.hash
                    print("Most obsolete vulnerable commit: ", older_vulnerable_commit.committer_date)

                    # Converting the generator of all commits in a list object, and then taking the index of the fixed
                    # commit in the list to obtain the commit before the fixed one
                    fixed_commit = repository.git.get_commit(fixed_hash)
                    commit_generator = repository.git.get_list_commits()
                    array_commit = list(commit_generator)
                    index_commit = array_commit.index(fixed_commit)
                    pre_fix = array_commit[index_commit - 1]
                    pre_fix_hash = pre_fix.hash

                    last_commit = repository.git.get_head()
                    index_last = array_commit.index(last_commit)
                    similar_commits = list()
                    print("Fixed commit files: ", fixed_commit.files)
                    for c in range(index_commit+1, index_last):
                        if fixed_commit.files-3 <= array_commit[c].files <= fixed_commit.files+3:
                            similar_commits.append(array_commit[c])

                    last_similar_commit = None
                    for c in similar_commits:
                        if last_similar_commit is None:
                            last_similar_commit = c
                        elif last_similar_commit.committer_date < c.committer_date:
                            last_similar_commit = c

                if vulnerable_hash is None or last_similar_commit is None or pre_fix_hash is None:
                    print("\nError during commit search\n")
                    continue
                else:
                    print("Fixed hash: ", fixed_hash)
                    print("Pre fix hash: ", pre_fix_hash)
                    print("Vulnerable hash: ", vulnerable_hash)
                    print("Last similar fixed commit hash: ", last_similar_commit.hash)

                    metric_calculation_and_writing(vulnerable_hash, pre_fix_hash, commit_link, writer, 1)
                    metric_calculation_and_writing(fixed_hash, last_similar_commit.hash, commit_link, writer, 0)
            except Exception as e:
                sys.stderr.write(str(e))
                continue


if __name__ == "__main__":
    main()
