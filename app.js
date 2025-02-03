// app.js

// Global state
let currentQuestion = null;
let solvedProblems = new Set();
let pyodideInstance = null;
let editor = null;
const TAB_ID = Date.now().toString();
let isActiveTab = false;

// Use a single, consistent key for localStorage
const STORAGE_KEY_STATE = 'bioLeetcodeState';         // For progress (solved problems, code)
const STORAGE_KEY_PROBLEMS = 'bio_leetcode_problems'; // For each problem's saved code
const STORAGE_KEY_ACTIVE_TAB = 'bio_leetcode_active_tab';

// ======================================
// 1. Initialize Pyodide
// ======================================
async function initPyodide() {
  if (!pyodideInstance) {
    pyodideInstance = await loadPyodide();
  }
  return pyodideInstance;
}

// ======================================
// 2. Tab Management
// ======================================
function claimActiveTab() {
  localStorage.setItem(STORAGE_KEY_ACTIVE_TAB, TAB_ID);
  isActiveTab = true;
}

function checkTabStatus() {
  const activeTab = localStorage.getItem(STORAGE_KEY_ACTIVE_TAB);
  const wasActive = isActiveTab;
  isActiveTab = (activeTab === TAB_ID);

  if (wasActive !== isActiveTab) {
    // Tab just went inactive
    if (!isActiveTab) {
      document.body.style.opacity = '0.5';
      editor.setOption('readOnly', true);

      const warning = document.createElement('div');
      warning.style.position = 'fixed';
      warning.style.top = '50%';
      warning.style.left = '50%';
      warning.style.transform = 'translate(-50%, -50%)';
      warning.style.backgroundColor = '#ff6b6b';
      warning.style.padding = '20px';
      warning.style.borderRadius = '5px';
      warning.style.color = 'white';
      warning.style.zIndex = '1000';
      warning.textContent = 'Please use the active tab instead. This tab is now inactive.';
      document.body.appendChild(warning);
    }
  }
}

// ======================================
// 3. React App Mounting
// ======================================
function mountReactApp() {
  const root = document.getElementById('reactRoot');
  ReactDOM.render(
    React.createElement(AppWrapper, {
      questions: questions,
      solvedProblems: Array.from(solvedProblems),
      onProblemSelect: loadQuestion,
      currentQuestion: currentQuestion
    }),
    root
  );
}

// ======================================
// 4. Progress Management
// ======================================
function saveProgress() {
  try {
    const state = {
      solvedProblems: Array.from(solvedProblems),
      currentProblem: currentQuestion
        ? {
            id: currentQuestion.id,
            code: editor.getValue()
          }
        : null,
      timestamp: Date.now()
    };
    localStorage.setItem(STORAGE_KEY_STATE, JSON.stringify(state));
    mountReactApp();
  } catch (e) {
    console.error(`Save error: ${e.message}`);
  }
}

function loadProgress() {
  try {
    const savedState = localStorage.getItem(STORAGE_KEY_STATE);
    if (!savedState) {
      return false;
    }
    const state = JSON.parse(savedState);
    // Restore solved problems
    solvedProblems = new Set(state.solvedProblems || []);
    // Restore current problem if exists
    if (state.currentProblem && state.currentProblem.id) {
      const problem = questions.find(q => q.id === state.currentProblem.id);
      if (problem) {
        currentQuestion = problem;
        editor.setValue(state.currentProblem.code || problem.starterCode);
        mountReactApp();
        return true;
      }
    }
    return false;
  } catch (e) {
    console.error(`Load error: ${e.message}`);
    return false;
  }
}

function resetProgress() {
  if (confirm('Are you sure you want to reset all progress? This cannot be undone.')) {
    localStorage.removeItem(STORAGE_KEY_STATE);
    solvedProblems = new Set();
    mountReactApp();
    selectRandomQuestion();
  }
}

// ======================================
// 5. Question Management
// ======================================
function saveProblemState() {
  if (!currentQuestion) return;
  try {
    const allProblems = JSON.parse(localStorage.getItem(STORAGE_KEY_PROBLEMS) || '{}');
    allProblems[currentQuestion.id] = editor.getValue();
    localStorage.setItem(STORAGE_KEY_PROBLEMS, JSON.stringify(allProblems));
  } catch (e) {
    console.error(`Error saving problem state: ${e.message}`);
  }
}

function loadQuestion(question) {
  if (!question) return;
  saveProblemState(); // Save old problem before switching
  currentQuestion = question;

  try {
    const allProblems = JSON.parse(localStorage.getItem(STORAGE_KEY_PROBLEMS) || '{}');
    const savedCode = allProblems[question.id];
    editor.setValue(savedCode || question.starterCode);
    mountReactApp();
  } catch (e) {
    editor.setValue(question.starterCode);
    console.error(`Error loading problem state: ${e.message}`);
  }
}

function selectRandomQuestion() {
  let unsolvedQuestions = questions.filter(q => !solvedProblems.has(q.id));
  if (unsolvedQuestions.length === 0) {
    unsolvedQuestions = questions;
  }
  const randomQuestion = unsolvedQuestions[Math.floor(Math.random() * unsolvedQuestions.length)];
  loadQuestion(randomQuestion);
}

// ======================================
// 6. Testing Logic with Pyodide
// ======================================
async function runTestCase(userCode, testCase) {
  const pyodide = await initPyodide();
  try {
    await pyodide.runPythonAsync(`
import sys
from io import StringIO
sys.stdout = StringIO()
`);
    await pyodide.runPythonAsync(userCode);
    await pyodide.runPythonAsync(testCase.code);

    let output = await pyodide.runPythonAsync('sys.stdout.getvalue()');
    return output.trim();
  } catch (error) {
    return 'Error: ' + error;
  }
}

async function checkAnswer() {
  let userCode = editor.getValue();
  const testResultsContainer = document.createElement('div');
  testResultsContainer.id = 'testResults';
  testResultsContainer.innerHTML = '<h3 class="font-semibold mb-3">Test Cases:</h3>';

  let allPassed = true;
  
  for (let i = 0; i < currentQuestion.testCases.length; i++) {
    const testCase = currentQuestion.testCases[i];
    const output = await runTestCase(userCode, testCase);

    const testDiv = document.createElement('div');
    testDiv.className = 'test-case';

    const passed = output === String(testCase.expected); // ensure string comparison
    allPassed = allPassed && passed;

    testDiv.classList.add(passed ? 'passed' : 'failed');
    testDiv.innerHTML = `
      <div class="font-medium mb-2">Test Case ${i + 1}</div>
      <div class="space-y-2">
          <div>Input: <code class="bg-gray-100 px-2 py-1 rounded">${testCase.input}</code></div>
          <div>Expected: <code class="bg-gray-100 px-2 py-1 rounded">${JSON.stringify(testCase.expected)}</code></div>
          ${
            passed
              ? '<div class="text-green-600">✓ Passed</div>'
              : `<div class="text-red-600">✗ Failed</div>
                 <div class="output-actual">Your output: ${output}</div>`
          }
      </div>
    `;
    testResultsContainer.appendChild(testDiv);
  }

  if (allPassed) {
    solvedProblems.add(currentQuestion.id);
    mountReactApp();
    saveProgress();
  }

  // Replace old results if exist
  const existingResults = document.getElementById('testResults');
  if (existingResults) {
    existingResults.replaceWith(testResultsContainer);
  } else {
    const editorSection = document.getElementById('editorSection');
    editorSection.appendChild(testResultsContainer);
  }

  // Overall result
  const resultMessage = document.createElement('div');
  resultMessage.id = 'resultMessage';
  resultMessage.className = 'card mt-4';
  resultMessage.innerHTML = `
    <h3 class="font-semibold mb-2">
      ${
        allPassed
          ? '<span class="text-green-600">✓ All Test Cases Passed!</span>'
          : '<span class="text-red-600">✗ Some Test Cases Failed</span>'
      }
    </h3>
    ${allPassed ? '' : '<p class="text-gray-600">Review the test cases below and try again.</p>'}
  `;

  testResultsContainer.insertBefore(resultMessage, testResultsContainer.firstChild);
}

// ======================================
// 7. Initialize CodeMirror Editor
// ======================================
function initializeEditor() {
  editor = CodeMirror.fromTextArea(document.getElementById('codeEditor'), {
    mode: 'python',
    lineNumbers: true,
    theme: 'default',
    viewportMargin: Infinity,
    lineWrapping: true,
    tabSize: 4,
    indentUnit: 4,
    extraKeys: {
      Tab: function(cm) {
        if (cm.somethingSelected()) {
          cm.indentSelection('add');
        } else {
          cm.replaceSelection('    ', 'end');
        }
      }
    }
  });

  editor.on('change', _.debounce(() => {
    if (currentQuestion) {
      saveProgress();
    }
  }, 1000));
}

// ======================================
// 8. Main Event Listeners
// ======================================
window.addEventListener('load', () => {
  initializeEditor();

  // Make sure we claim active tab if none is set
  if (!localStorage.getItem(STORAGE_KEY_ACTIVE_TAB)) {
    claimActiveTab();
  }
  checkTabStatus();
  setInterval(checkTabStatus, 1000);

  // Load progress or select a random question
  if (!loadProgress()) {
    selectRandomQuestion();
  }
  mountReactApp();
});

window.addEventListener('focus', () => {
  claimActiveTab();
});

// NOTE: Updated key to match STORAGE_KEY_STATE
window.addEventListener('storage', (e) => {
  if (e.key === STORAGE_KEY_STATE) {
    try {
      const newState = JSON.parse(e.newValue);
      if (!newState || !newState.timestamp) return;

      solvedProblems = new Set(newState.solvedProblems || []);
      mountReactApp();

      if (newState.currentProblem && newState.currentProblem.id) {
        const problem = questions.find(q => q.id === newState.currentProblem.id);
        if (problem && (!currentQuestion || currentQuestion.id !== newState.currentProblem.id)) {
          loadQuestion(problem);
        }
      }
    } catch (err) {
      console.error(`Error processing storage event: ${err.message}`);
    }
  }
});
