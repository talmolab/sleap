# Contributing to SLEAP

As our community grows it is important to adhere to a set of contribution guidelines. These guidelines may change as needed. Please feel free to propose changes in a pull request! 

#### Table Of Contents

1) [Code of Conduct](#code-of-conduct)
2) [Contributing](#contributing)
    * [Issues](#issues)
    * [Discussions](#discussions)
    * [Pull Requests](#pull-requests)
3) [Style Guides](#style-guides)

## Code of Conduct

Everyone contributing to SLEAP is governed by the [SLEAP Code of Conduct](CODE_OF_CONDUCT.md). As such, all are expected to abide by this code. Please report any unacceptable behavior to talmo@salk.edu.

## Contributing

Github has made it easy to separate issues from discussions. Generally speaking, if something warrants a pull request (e.g. bug fixes, TODOs) it should be submitted as an issue. Conversely, general questions about usage and improvement ideas should go into discussions. These both act as staging areas before adding the items to the SLEAP Roadmap. Once added to the Roadmap, items can be given priority, status, assignee(s), etc. 

### Issues

* Check [open/closed issues](https://github.com/talmolab/sleap/issues), [ideas](https://github.com/talmolab/sleap/discussions/categories/ideas), and [the help page](https://github.com/talmolab/sleap/discussions/categories/help) to make sure issue doesn't already exist / has been solved.
* Create new issue using the [issue template](https://github.com/talmolab/sleap/blob/arlo/contributing_guide/.github/ISSUE_TEMPLATE/bug_report.md).

### Discussions

* This is a place to go to ask questions and propose new ideas.
* 3 categories: Help, Ideas, General
   * **Help** - Having trouble with software, user experience issue, etc.
   * **Ideas** - Enhancements, things that would make the user experience nicer but not necessarily a problem with the code.
   * **General** - If it doesn't fall into help/ideas it goes here as long as it isn't bug fix (issue).

### Pull Requests

1) Install source code [develop branch](https://sleap.ai/installation.html#conda-from-source) and follow instructions to create conda env, etc.
2) Create fork.
   * Either work on develop branch in own fork or can create branch on either github or cli.
   * If creating a branch, use your name followed by a relevant keyword for your changes, eg: `git checkout -b john/some_issue`
3) Make some changes to source code or additions that tackle the issue(s).
4) Write [tests](https://github.com/talmolab/sleap/tree/develop/tests).
   * Can either write before PR or submit draft PR and then write tests to narrow down error prone lines.
   * The test(s) should go into relevant subtest folders to the proposed change(s).
   * Test(s) should aim to hit every point in the proposed change(s) - cover edge cases to best of your ability.
   * Try to hit code coverage points.
5) Add files and commit (make sure to correct branch!).
6) Create Draft PR (on github - follow instructions in template).
   * Make sure tests pass and code coverage is good.
   * If tests fail, repeat steps 4-7.
7) Once draft pr looks good, submit a PR (hit `ready for review` button).
   * Optionally assign a reviewer on right of screen.
8) If reviewer requests changes, repeat steps 4-8.
9) Once reviewer signs off they will squash + merge.

## Style Guides

* **Linting** - [black](https://black.readthedocs.io/en/stable/) version 21.6b0 (see [dev_requirements](https://github.com/talmolab/sleap/blob/develop/dev_requirements.txt) for any changes).
* **Coding** - generally follow pep8, type hinting is encouraged.
* Use Google style [docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) to document code.

#### Thank you for contributing to SLEAP! 
:heart:
