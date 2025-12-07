import re


def recursive_splitting(text: str, size: int, overlap: int) -> list[str]:
    separators = ["\n\n", "\n", " ", ""]
    is_separator_regex = False
    keep_separator = True  # Equivalent to 'start'
    length_function = len
    strip_whitespace = True
    chunk_size = size
    chunk_overlap = overlap

    def _split_text_with_regex(
        text: str, separator: str, keep_separator: bool
    ) -> list[str]:
        if not separator:
            return [text]
        if not keep_separator:
            return re.split(separator, text)
        _splits = re.split(f"({separator})", text)
        splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
        if len(_splits) % 2 == 0:
            splits += [_splits[-1]]
        return [_splits[0]] + splits if _splits[0] else splits

    def _join_docs(docs: list[str], separator: str) -> str | None:
        text = separator.join(docs)
        if strip_whitespace:
            text = text.strip()
        return text if text else None

    def _merge_splits(splits: list[str], separator: str) -> list[str]:
        separator_len = length_function(separator)
        docs = []
        current_doc = []
        total = 0
        for d in splits:
            len_ = length_function(d)
            if total + len_ + (separator_len if current_doc else 0) > chunk_size:
                if total > chunk_size:
                    pass  # Optionally add warning
                if current_doc:
                    doc = _join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while total > chunk_overlap or (
                        total + len_ + (separator_len if current_doc else 0)
                        > chunk_size
                        and total > 0
                    ):
                        total -= length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += len_ + (separator_len if len(current_doc) > 1 else 0)
        doc = _join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _split_text(text: str, separators: list[str]) -> list[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            separator_ = _s if is_separator_regex else re.escape(_s)
            if not _s:
                separator = _s
                break
            if re.search(separator_, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break
        separator_ = separator if is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, separator_, keep_separator)
        _separator = "" if keep_separator else separator
        good_splits = []
        for s in splits:
            if length_function(s) < chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = _merge_splits(good_splits, _separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = _split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if good_splits:
            merged_text = _merge_splits(good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    return _split_text(text, separators)


if __name__ == "__main__":
    text = """Charles James Kirk was born on October 14, 1993, in the Chicago suburb of Arlington Heights, Illinois,[1] and raised in nearby Prospect Heights.[2] His father, Robert W. Kirk, is an architect who was involved in the construction of Trump Tower.[3][4] His mother, Kathryn (née Smith),[5] is a former trader at the Chicago Mercantile Exchange who subsequently worked as a mental health counselor.[2][3][4] He had one sibling, a younger sister, Mary, who went on to become an art curator in Chicago.[6][7]

Kirk described his parents as moderate Republicans.[2] They were active in conservative circles, and his father was a major donor to the Mitt Romney 2012 presidential campaign.[1] Raised in the Presbyterian Church, Kirk was a member of the Boy Scouts of America and earned the rank of Eagle Scout.[8][9] He experienced a political awakening in middle school, during which he read books by economist Milton Friedman and became more attracted to the principles of the Republican Party.[2]

In 2010, during his junior year at Wheeling High School, Kirk volunteered for the successful U.S. Senate campaign of Illinois Republican Mark Kirk (no relation).[10] Also during his junior year, he began listening to The Rush Limbaugh Show, a prominent conservative talk radio broadcast.[1] In his senior year, he initiated a boycott of cookies at the school's cafeteria to reverse a price increase.[2] He also wrote an essay for Breitbart News alleging liberal bias in high-school textbooks; it led to his first media appearance on Fox Business at age 17.[11][12]

In 2012, Kirk applied to West Point but was rejected.[11][12] Although he was accepted that same year to Baylor University in Waco, Texas, he enrolled instead at Harper College, a community college in Palatine, Illinois. Withdrawing after one semester, he left Harper to concentrate on his work with Turning Point USA, the political group he co-founded with conservative businessman and mentor Bill Montgomery.[1][11] In 2015, Kirk enrolled part-time at King's College in New York City, taking online classes.[13] Kirk did not receive a college degree during his lifetime, a fact he noted in debates with academics and students.[14]
In May 2012, Kirk gave a speech at Benedictine University's "Youth Government Day", where he met Bill Montgomery, a 72-year-old retiree who was then a Tea Party–backed legislative candidate.[15][16] Montgomery later said that the speakers at the event had bored the audience of a few hundred high-school kids, but they began to pay rapt attention when Kirk started speaking. Montgomery then encouraged Kirk to pursue political activism full-time.[17][1] A month after they first met, Montgomery and Kirk co-founded Turning Point USA, wanting to start an organization rivaling liberal groups such as MoveOn.org.[18][15] Kirk described it as a student organization advocating for free markets and limited government.[19] At the 2012 Republican National Convention, Kirk met Foster Friess, a former investment manager and prominent Republican donor, and persuaded him to finance the organization.[15][18]

Kirk remained the executive director, chief fundraiser, and the public face of Turning Point USA until his death in 2025.[20][8][19] He became known for visiting college campuses to debate with ideological opponents, typically students, and persuade them to consider conservative candidates.[21] According to the Associated Press, video clips of Kirk's campus appearances spread online, helping him "secure a steady stream of donations that transformed Turning Point into one of the country's largest political organizations".[19] Turning Point eventually began holding massive rallies in which top conservative leaders addressed tens of thousands of young voters.[19] In 2025, TPUSA said it had chapters at more than 2,000 college and high school campuses, and that it had received 32,000 inquiries about starting new chapters in the days after Kirk's death.[22]

TPUSA's activities include publication of the Professor Watchlist and the School Board Watchlist.[23] Critics of these watchlists say they threaten academic freedom and have led to the targeted harassment of academics.[24][25] In 2019, the Professor Watchlist was briefly suspended by its web host.[26] In 2020, ProPublica investigated TPUSA's finances and found that the organization made "misleading financial claims", that the audits were not done by an independent auditor, and that the leaders had enriched themselves while advocating for Trump. ProPublica also reported that Kirk's salary from TPUSA had increased from $27,000 to nearly $300,000 and that he had bought an $855,000 condo in Longboat Key, Florida.[27] In 2020, Turning Point USA had $39.2 million in revenue.[28] Kirk earned a salary of more than $325,000 from TPUSA and related organizations.[29]

Turning Point Academy
In 2021, TPUSA announced it would launch an online academy as an alternative to schools "poisoning our youth with anti-American ideas". Turning Point Academy was intended to cater to families seeking an "America-first education".[30] Arizona education firm StrongMind initially partnered with TPUSA with plans to open the academy by the fall of 2022 and assessed its "potential to generate over $40 million in gross revenue at full capacity (10,000 students)".[30] The partnership ended after StrongMind received backlash from its own employees, and key subcontractor Freedom Learning Group, which prepared course content for the academy, also backed out.[30] In 2022, Turning Point partnered with Dream City Christian School, a private school that has campuses in Glendale and Scottsdale, Arizona, and is affiliated with Dream City Church.[31][32] In the 2022–2023 school year, the school received $900,000 in Arizona school voucher funds.[31]

Turning Point Action
Main article: Turning Point Action
In May 2019, it was reported that Kirk was preparing to launch Turning Point Action, a 501(c)(4) entity designed to elect more conservatives.[33] In July 2019, Kirk announced that Turning Point Action had acquired Students for Trump along with "all associated media assets".[34] He became chairman and launched a campaign to mobilize the youth vote for the 2020 Trump reelection campaign.[11] The unsuccessful effort led TPUSA and the 2020 Trump campaign to blame each other for an overall decline in Trump's youth support.[35] In December 2022, Kirk announced the Mount Vernon Project, an initiative by Turning Point Action to remove members from the Republican National Committee who were not "grassroot conservatives".[36]

On January 5, 2021, the day before the Washington, D.C., protest that led to the January 6 U.S. Capitol attack, Kirk wrote on Twitter that Turning Point Action and Students for Trump were sending more than 80 "buses of patriots to D.C. to fight for this president".[37][38] A spokesman for Turning Point said that the groups ended up sending seven buses, not 80, with 350 students.[37][39] In the lead-up to the storming, Kirk said he was "getting 500 emails a minute calling for a civil war".[40] Publix heiress Julie Fancelli gave Kirk's organizations $1.25 million to fund the buses to the January 6 event. Kirk also paid $60,000 for Kimberly Guilfoyle to speak at the rally.[41]

Afterward, Kirk said the violent acts at the Capitol were not an insurrection and did not represent mainstream Trump supporters.[42][43] Appearing before the U.S. House Select Committee on the January 6 Attack in December 2022, he pleaded the Fifth Amendment privilege against self-incrimination. His team provided the committee "with 8,000 pages of records in response to its requests".[44] In another closed-door meeting of the House January 6 Committee, Ali Alexander blamed Kirk and TPUSA for financing the travel of demonstrators to the Save America rally.[45] TPUSA spokesperson Andrew Kolvet denied that Kirk advocated for violence and gave a statement saying "Charlie wants to save America with words, persuasion, courage and common sense. The left is desperate to conjure up some Christian bogeyman that simply doesn't exist. We're telling churches: Either get involved and have a say in the direction of your country or you'll leave a void that someone else who doesn't share your values will fill."[46]
Falkirk Center for Faith and Liberty

In November 2019, Kirk and Jerry Falwell Jr. co-founded the Falkirk Center for Faith and Liberty, a right-wing think tank funded, owned, and housed by Liberty University.[47][48] "Falkirk" was a portmanteau of "Falwell" and "Kirk".[48] Fellows included Antonia Okafor, director of outreach for Gun Owners of America; Sebastian Gorka, former deputy assistant to Trump; and Jenna Ellis, a senior legal counselor for Trump.[49]

In 2020, the Falkirk Center spent at least $50,000 on Facebook advertisements promoting Trump and Republican candidates.[50] Students and alumni raised objections to the organization's aggressive political tone, which they considered inconsistent with the university's mission.[48] Falwell resigned as president of Liberty University in August 2020, and the university did not renew Kirk's one-year contract in late 2020. In 2021, the university renamed the organization Standing for Freedom Center.[48]
Turning Point Faith
After Liberty University did not renew Kirk's contract with the Falkirk Center for Faith and Liberty in 2021, Kirk and Pentecostal pastor Rob McCoy founded Turning Point Faith, an organization that encouraged pastors and other church leaders to be active in local and national political issues.[48][51] Its activities include faith-based voter drives and promotion of TPUSA's views, with the stated goal to help churches become more civically engaged so that American society can "return to foundational Christian values".[52] According to TPUSA's 2021 Investor Prospectus, the program—with a budget of $6.4 million—"will 'address America's crumbling religious foundation by engaging thousands of pastors nationwide' in order to 'breathe renewed civic engagement into our churches'".[53]"""
    print(recursive_splitting(text, 1200, 0))
    print(len(recursive_splitting(text, 1200, 0)))
