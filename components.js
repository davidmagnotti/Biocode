// QuestionDisplay Component with mobile-first design
const QuestionDisplay = ({ question }) => {
    if (!question) {
        return null;
    }

    const title = question.title;
    const description = question.description;
    const example = question.example;

    return React.createElement('div', { 
        className: 'card' 
    }, [
        // Compact header
        React.createElement('div', { 
            key: 'header',
            className: 'mb-3' 
        }, [
            React.createElement('h2', { 
                className: 'text-lg font-semibold' 
            }, title),
            React.createElement('p', { 
                className: 'text-sm text-gray-600 mt-1' 
            }, description)
        ]),

        // Compact example
        React.createElement('div', { 
            key: 'example',
            className: 'bg-gray-50 rounded-xl p-3 mb-3' 
        }, [
            React.createElement('h3', { 
                className: 'text-xs font-medium text-gray-700 mb-1' 
            }, 'Example'),
            React.createElement('code', { 
                className: 'text-sm block' 
            }, example)
        ]),

        // Compact function signature
        React.createElement('div', { 
            key: 'signature',
            className: 'pt-3 border-t border-gray-200' 
        }, [
            React.createElement('h3', { 
                className: 'text-xs font-medium text-gray-700 mb-1' 
            }, 'Function Signature'),
            React.createElement('code', { 
                className: 'text-sm block' 
            }, question.starterCode.split('\n')[0])
        ])
    ]);
};

// Updated ProblemNavigator for mobile
const ProblemNavigator = ({ problems, solvedProblems, onProblemSelect }) => {
    const [searchTerm, setSearchTerm] = React.useState('');
    const [currentPage, setCurrentPage] = React.useState(1);
    const problemsPerPage = 20; // Show more problems on mobile grid

    const filteredProblems = problems.filter(problem => {
        const searchLower = searchTerm.toLowerCase();
        return problem.description.toLowerCase().includes(searchLower);
    });

    const totalPages = Math.ceil(filteredProblems.length / problemsPerPage);
    const startIndex = (currentPage - 1) * problemsPerPage;
    const displayedProblems = filteredProblems.slice(startIndex, startIndex + problemsPerPage);
    const totalSolved = solvedProblems.size;
    const completionPercentage = Math.round((totalSolved / problems.length) * 100);

    return React.createElement('div', { 
        className: 'card' 
    }, [
        // Compact progress section
        React.createElement('div', { 
            key: 'progress',
            className: 'mb-3' 
        }, [
            React.createElement('div', { 
                className: 'flex justify-between items-center mb-2' 
            }, [
                React.createElement('span', { 
                    className: 'text-sm font-medium' 
                }, `${totalSolved} / ${problems.length} Complete`),
                React.createElement('span', { 
                    className: 'text-sm text-gray-600' 
                }, `${completionPercentage}%`)
            ]),
            React.createElement('div', { 
                className: 'progress-bar' 
            }, [
                React.createElement('div', {
                    className: 'progress-fill',
                    style: { width: `${completionPercentage}%` }
                })
            ])
        ]),

        // Compact search
        React.createElement('input', {
            key: 'search',
            type: 'search',
            placeholder: 'Search problems...',
            value: searchTerm,
            onChange: (e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
            },
            className: 'search-input mb-3'
        }),

        // Mobile-friendly grid
        React.createElement('div', { 
            key: 'grid',
            className: 'problem-grid' 
        }, displayedProblems.map((problem, index) => {
            const problemNumber = startIndex + index + 1;
            const isSolved = solvedProblems.has(problem.id);
            
            return React.createElement('button', {
                key: problem.id,
                onClick: () => onProblemSelect(problem),
                className: `problem-item ${isSolved ? 'bg-green-50 border-green-200' : 'bg-white border-gray-200'}`
            }, [
                React.createElement('span', {
                    key: 'number',
                    className: isSolved ? 'text-green-600' : 'text-gray-600'
                }, problemNumber),
                React.createElement('span', {
                    key: 'icon',
                    className: `text-xs ${isSolved ? 'text-green-500' : 'text-gray-400'}`
                }, isSolved ? '✓' : '')
            ]);
        }))
    ]);
};

// Main App Wrapper Component
const AppWrapper = ({ questions, solvedProblems, onProblemSelect, currentQuestion }) => {
    return React.createElement('div', { className: 'w-full space-y-6' }, [
        React.createElement('h1', { 
            key: 'title',
            className: 'text-2xl font-bold text-center text-gray-900'
        }, 'Biocode'),
        React.createElement(ProblemNavigator, {
            key: 'navigator',
            problems: questions,
            solvedProblems: new Set(solvedProblems),
            onProblemSelect: onProblemSelect
        }),
        React.createElement(QuestionDisplay, {
            key: 'question',
            question: currentQuestion
        })
    ]);
};