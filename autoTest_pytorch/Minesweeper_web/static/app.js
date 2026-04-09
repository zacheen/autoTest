const difficultyGroup = document.getElementById("difficulty-group");
const boardElement = document.getElementById("board");
const mineCounterElement = document.getElementById("mine-counter");
const timerElement = document.getElementById("timer");
const statusElement = document.getElementById("status-text");
const newGameButton = document.getElementById("new-game-button");

const cellSymbols = {
    flagged: "F",
    mine: "X",
    hit_mine: "X",
    flagged_mine: "F",
    wrong_flag: "x",
};

let availableDifficulties = {};
let selectedDifficulty = "Beginner";
let currentGame = null;
let timerIntervalId = null;
let localElapsed = 0;
let elapsedAnchor = Date.now();
const difficultyOrder = ["Training 6x6", "Beginner", "Intermediate", "Expert"];

function formatCounter(value) {
    return String(Math.max(0, value)).padStart(3, "0");
}

function syncTimer(state) {
    localElapsed = state.elapsed_seconds;
    elapsedAnchor = Date.now();
    updateTimerDisplay(state.status);
}

function getDisplayedElapsed(status) {
    if (status === "playing") {
        return localElapsed + Math.floor((Date.now() - elapsedAnchor) / 1000);
    }
    return localElapsed;
}

function updateTimerDisplay(status) {
    timerElement.textContent = formatCounter(getDisplayedElapsed(status));
}

function startTimerTicking() {
    if (timerIntervalId !== null) {
        window.clearInterval(timerIntervalId);
    }

    timerIntervalId = window.setInterval(() => {
        if (currentGame) {
            updateTimerDisplay(currentGame.status);
        }
    }, 250);
}

function updateStatusText(state) {
    const labels = {
        ready: "Ready",
        playing: "Playing",
        won: "You won",
        lost: "Game Over",
    };
    statusElement.textContent = labels[state.status] || state.status;
}

function applyBoardSizing(rows, cols) {
    const stage = document.querySelector(".board-shell");
    const rect = stage.getBoundingClientRect();
    const horizontalPadding = 28;
    const verticalPadding = 28;
    const cellWidth = (rect.width - horizontalPadding - (cols - 1) * 2) / cols;
    const cellHeight = (rect.height - verticalPadding - (rows - 1) * 2) / rows;
    const size = Math.max(24, Math.min(46, Math.floor(Math.min(cellWidth, cellHeight))));
    document.documentElement.style.setProperty("--cell-size", `${size}px`);
}

function renderDifficultyButtons() {
    difficultyGroup.innerHTML = "";

    const orderedDifficulties = difficultyOrder.filter((difficultyName) => difficultyName in availableDifficulties);

    orderedDifficulties.forEach((difficultyName) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "action-button difficulty-button";
        button.textContent = difficultyName;
        button.classList.add(`difficulty-${difficultyName.toLowerCase().replace(/\s+/g, "-")}`);

        button.addEventListener("click", () => {
            selectedDifficulty = difficultyName;
            createGame(difficultyName);
        });

        difficultyGroup.appendChild(button);
    });
}

function buildCellButton(cell, row, col) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "cell";
    button.dataset.row = String(row);
    button.dataset.col = String(col);
    button.classList.add(cell.state.replace("_", "-"));

    if (cell.state === "revealed") {
        button.disabled = true;
        if (cell.value > 0) {
            button.textContent = String(cell.value);
            button.classList.add(`n${cell.value}`);
        }
    } else if (cellSymbols[cell.state]) {
        button.textContent = cellSymbols[cell.state];
        if (cell.state !== "flagged") {
            button.disabled = true;
        }
    }

    if (currentGame.status === "won" || currentGame.status === "lost") {
        button.disabled = true;
    }

    button.addEventListener("click", async () => {
        if (currentGame.status === "won" || currentGame.status === "lost") {
            return;
        }
        await actOnCell("click", row, col);
    });

    button.addEventListener("contextmenu", async (event) => {
        event.preventDefault();
        if (currentGame.status === "won" || currentGame.status === "lost") {
            return;
        }
        await actOnCell("flag", row, col);
    });

    return button;
}

function renderBoard(state) {
    currentGame = state;
    boardElement.innerHTML = "";
    boardElement.style.gridTemplateColumns = `repeat(${state.cols}, var(--cell-size))`;
    boardElement.style.gridTemplateRows = `repeat(${state.rows}, var(--cell-size))`;

    state.board.forEach((rowCells, rowIndex) => {
        rowCells.forEach((cell, colIndex) => {
            boardElement.appendChild(buildCellButton(cell, rowIndex, colIndex));
        });
    });

    mineCounterElement.textContent = formatCounter(state.remaining_mines);
    updateStatusText(state);
    syncTimer(state);
    renderDifficultyButtons();
    applyBoardSizing(state.rows, state.cols);
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, {
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {}),
        },
        ...options,
    });

    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || "Request failed");
    }
    return data;
}

async function fetchState(gameId) {
    const state = await requestJson(`/api/games/${gameId}`);
    renderBoard(state);
    return state;
}

async function createGame(difficulty) {
    const state = await requestJson("/api/games", {
        method: "POST",
        body: JSON.stringify({ difficulty }),
    });

    selectedDifficulty = state.difficulty;
    window.localStorage.setItem("minesweeper-web-game-id", state.game_id);
    renderBoard(state);
}

async function actOnCell(action, row, col) {
    const state = await requestJson(`/api/games/${currentGame.game_id}/${action}`, {
        method: "POST",
        body: JSON.stringify({ row, col }),
    });

    renderBoard(state);
}

async function loadDifficulties() {
    const data = await requestJson("/api/difficulties");
    availableDifficulties = data.difficulties;
    renderDifficultyButtons();
}

async function bootstrap() {
    startTimerTicking();
    await loadDifficulties();

    const storedGameId = window.localStorage.getItem("minesweeper-web-game-id");
    if (storedGameId) {
        try {
            const state = await fetchState(storedGameId);
            selectedDifficulty = state.difficulty;
            renderDifficultyButtons();
            return;
        } catch (error) {
            window.localStorage.removeItem("minesweeper-web-game-id");
        }
    }

    await createGame(selectedDifficulty);
}

newGameButton.addEventListener("click", async () => {
    await createGame(selectedDifficulty);
});

window.addEventListener("resize", () => {
    if (currentGame) {
        applyBoardSizing(currentGame.rows, currentGame.cols);
    }
});

bootstrap().catch((error) => {
    statusElement.textContent = error.message;
});
