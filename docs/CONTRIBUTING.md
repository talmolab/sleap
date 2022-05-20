# Contributing to SLEAP

Thank you for contributing to SLEAP!

As our community grows it is important to adhere to a set of contribution guidelines. These guidelines may change as needed. Please feel free to propose changes in a pull request! 

#### Table Of Contents

1) [Code of Conduct](#code-of-conduct)
2) [Contributing](#contributing)
    * [Issues](#issues)
    * [Discussions](#discussions)
    * [Pull Requests](#pull-requests)
3) [Style Guides](#style-guides)
4) [Miscellaneous](#miscellaneous)

## Code of Conduct

Everyone contributing to SLEAP is governed by the [SLEAP Code of Conduct](CODE_OF_CONDUCT.md). As such, all are expected to abide by this code. Please report any unacceptable behavior to talmo@salk.edu.

## Contributing

Github has made it easy to separate issues from discussions. Generally speaking, if something warrants a pull request (e.g. bug fixes, TODOs) it should be submitted as an issue. Conversely, general questions about usage and improvement ideas should go into discussions. These both act as staging areas before adding the items to the SLEAP Roadmap. Once added to the Roadmap, items can be given priority, status, assignee(s), etc. 

### Issues

* Check [open/closed issues](https://github.com/talmolab/sleap/issues), [discussions](https://github.com/talmolab/sleap/discussions), and [sleap.ai help page](https://sleap.ai/help.html) to make sure issue doesn't already exist / has been solved
* Create new issue using the [issue template](https://github.com/talmolab/sleap/blob/arlo/contributing_guide/docs/ISSUE_TEMPLATE.md)

### Discussions

* This is a place to go to ask questions and propose new ideas
* 3 categories: general, help, ideas
   * help - having trouble with software, user experience issue, etc
   * ideas - enhancements, things that would make the user experience nicer but not necessarily a problem with the code
   * general - if it doesn't fall into help/ideas it goes here as long as it isn't bug fix (issue)

### Pull Requests

1) install source code develop branch: https://sleap.ai/installation.html#conda-from-source
2) follow instructions to create conda env etc
3) create fork
   * either work on develop branch in own fork or can create branch on either github or cli
   * if creating a branch, use your name followed by a relevant keyword for your changes, eg: `git checkout -b john/some_issue`
4) make some changes to source code or additions that tackle the issue(s)
5) write [tests](https://github.com/talmolab/sleap/tree/develop/tests)
   * can either write before pr or submit draft pr and then write tests to narrow down error prone lines 
   * the test(s) should go into relevant subtest folders to the proposed change(s)
   * test(s) should aim to hit every point in the proposed change(s) - cover edge cases to best of your ability
   * try to hit code coverage points
6) add files and commit (make sure to correct branch!)
7) create Draft PR (on github - follow instructions in template)
   * make sure tests pass and code coverage is good
   * if tests fail, repeat steps 4-7
8) once draft pr looks good, submit a PR (hit `ready for review` button)
   *  optionally assign a reviewer on right of screen
9) if reviewer requests changes, repeat steps 4-8
10) once reviewer signs off they will squash + merge

## Style Guides

* linting - [black](https://black.readthedocs.io/en/stable/) version 21.6b0 (see [dev_requirements](https://github.com/talmolab/sleap/blob/develop/dev_requirements.txt) for any changes)
* coding - generally follow pep8, type hinting is encouraged

#### Thank you for contributing to SLEAP! 
:heart:
