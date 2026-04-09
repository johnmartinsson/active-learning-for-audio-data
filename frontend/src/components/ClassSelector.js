import React from 'react';

const ClassSelector = ({ x, y, classes, onSelect, onNewClass }) => {
  const style = {
    position: 'fixed',
    left: x,
    top: y,
    background: 'white',
    border: '1px solid #ccc',
    borderRadius: '6px',
    padding: '6px',
    zIndex: 1000,
    minWidth: '140px',
    boxShadow: '0 6px 16px rgba(0, 0, 0, 0.15)'
  };

  return (
    <div style={style}>
      {classes.map((c) => (
        <div
          key={c}
          onClick={() => onSelect(c)}
          style={{ cursor: 'pointer', padding: '4px 6px' }}
        >
          {c}
        </div>
      ))}

      <hr />

      <div
        onClick={onNewClass}
        style={{ cursor: 'pointer', padding: '4px 6px' }}
      >
        + New class
      </div>
    </div>
  );
};

export default ClassSelector;