% Determine learning rate based on chosen schedule
function lr = LRSchedule(Init, DropFrac, Tepoch, igen, type)
switch type
    case 'none'
        lr = Init;
    case 'step'
        lr = Init*exp(-DropFrac*floor(igen/Tepoch));
    case 'piece-wise'
        lr = Init*((DropFrac)^(floor(igen/Tepoch)));
end